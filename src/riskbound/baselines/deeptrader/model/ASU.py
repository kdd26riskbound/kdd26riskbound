import os
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS    = 1e-8
class nconv(nn.Module):
    def forward(self, x, A):
        if A.device != x.device:
            A = A.to(x.device)
        return torch.einsum("bcvl,vw->bcwl", x, A).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)
    def forward(self, x): return self.mlp(x)

class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super().__init__()
        self.nconv = nconv()
        c_mul      = (order * support_len + 1) * c_in
        self.mlp   = linear(c_mul, c_out)
        self.order = order
        self.drop  = dropout

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a); out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, a); out.append(x1)
        h = torch.cat(out, dim=1)
        return F.dropout(self.mlp(h), self.drop, self.training)

class SpatialAttentionLayer(nn.Module):
    def __init__(self, num_nodes: int, in_feat: int, win_len: int):
        super().__init__()
        self.L_fixed = win_len
        self.W1 = nn.Linear(win_len,        1,  bias=False)
        self.W2 = nn.Linear(in_feat,  win_len,  bias=False)
        self.W3 = nn.Linear(in_feat,        1,  bias=False)
        self.V  = nn.Linear(num_nodes,  num_nodes,  bias=False)

        self.bn1 = nn.BatchNorm1d(num_nodes)
        self.bn2 = nn.BatchNorm1d(num_nodes)
        self.bn3 = nn.BatchNorm1d(num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, N, L = x.shape

        if L < self.L_fixed:
            x = F.pad(x, (self.L_fixed - L, 0, 0, 0))
        elif L > self.L_fixed:
            x = x[..., -self.L_fixed:]

        p1 = x.permute(0, 2, 1, 3)
        p2 = x.permute(0, 2, 3, 1)

        p1 = self.bn1(self.W1(p1).squeeze(-1))
        p1 = self.bn2(self.W2(p1))
        p2 = self.bn3(self.W3(p2).squeeze(-1)).permute(0, 2, 1)

        S  = torch.softmax(self.V(torch.relu(torch.bmm(p1, p2))), dim=-1)
        return S

class SAGCN(nn.Module):
    def __init__(self,
                 num_nodes, in_features, hidden_dim, window_len,
                 dropout=0.3, kernel_size=2, layers=4,
                 supports=None, spatial_bool=True,
                 addaptiveadj=True, aptinit=None):
        super().__init__()
        self.layers           = layers
        self.gcn_bool         = spatial_bool
        self.sattn_bool       = spatial_bool
        self.addaptiveadj     = addaptiveadj
        self.supports         = supports

        self.start_conv = nn.Conv2d(in_features, hidden_dim, kernel_size=(1, 1))
        self.bn_start   = nn.BatchNorm2d(hidden_dim)

        self.tcns, self.res_convs = nn.ModuleList(), nn.ModuleList()
        self.bns,  self.gcns, self.sans = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        self.sup_len = len(supports) if supports is not None else 0
        if self.gcn_bool and addaptiveadj:
            self.nodevec = nn.Parameter(torch.randn(num_nodes, 1))
            self.sup_len += 1

        receptive_field = 1
        dil, add_scope = 1, kernel_size - 1
        for li in range(layers):
            self.tcns.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim,
                              kernel_size=(1, kernel_size), dilation=(1, dil)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm2d(hidden_dim)
                )
            )
            self.res_convs.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            self.bns.append(nn.BatchNorm2d(hidden_dim))

            if self.gcn_bool:
                self.gcns.append(
                    GraphConvNet(hidden_dim, hidden_dim, dropout,
                                 support_len=self.sup_len)
                )

            if self.sattn_bool:
                self.sans.append(
                    SpatialAttentionLayer(num_nodes, hidden_dim,
                                          receptive_field + add_scope)
                )

            dil            *= 2
            receptive_field += add_scope
            add_scope      *= 2

        self.receptive_field = receptive_field

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        if X.size(3) < self.receptive_field:
            X = F.pad(X, (self.receptive_field - X.size(3), 0, 0, 0))

        x = self.bn_start(self.start_conv(X))

        sup = self.supports
        if self.gcn_bool and self.addaptiveadj and sup is not None:
            adp = torch.softmax(torch.relu(self.nodevec @ self.nodevec.T), dim=0)
            sup = sup + [adp]

        for i in range(self.layers):
            res = self.res_convs[i](x)
            x   = self.tcns[i](x)
            if self.gcn_bool and sup is not None:
                x = self.gcns[i](x, sup)
            if self.sattn_bool:
                S = self.sans[i](x)
                x = torch.einsum("bnm,bfml->bfnl", S, x)

            x = self.bns[i](x + res[..., -x.size(3):])

        x = x[..., -1]
        return x.permute(0, 2, 1).contiguous()

class LiteTCN(nn.Module):
    def __init__(self, in_feat, hid, n_layers, k_size=2, dropout=0.4):
        super().__init__()
        self.rf = 1 + (k_size - 1) * (2 ** n_layers - 1)
        self.start = nn.Conv1d(in_feat, hid, 1)
        self.tcns, self.bns = nn.ModuleList(), nn.ModuleList()
        dil = 1
        for _ in range(n_layers):
            self.tcns.append(nn.Conv1d(hid, hid, k_size, dilation=dil))
            self.bns.append(nn.BatchNorm1d(hid)); dil *= 2
        self.end = nn.Conv1d(hid, 1, 1)

    def forward(self, X):
        X = X.permute(0, 2, 1)
        if X.size(2) < self.rf:
            X = F.pad(X, (self.rf - X.size(2), 0))
        h = self.start(X)
        for conv, bn in zip(self.tcns, self.bns):
            res = h; h = bn(conv(h)); h = h + res[..., -h.size(2):]
        return torch.sigmoid(self.end(h).squeeze(-1))

class ASU(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, window_len,
                 dropout=0.3, kernel_size=2, layers=4, supports=None,
                 spatial_bool=True, addaptiveadj=True, aptinit=None):

        super().__init__()
        self.sagcn = SAGCN(num_nodes, in_features, hidden_dim, window_len,
                           dropout, kernel_size, layers, supports,
                           spatial_bool, addaptiveadj, aptinit)
        self.bn1      = nn.BatchNorm1d(num_nodes)
        self.linear1  = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sagcn(inputs)
        x = self.bn1(x)
        logits = self.linear1(x).squeeze(-1)
        score  = torch.sigmoid(logits)

        minus_inf = torch.full_like(score, float("-inf"))
        score     = torch.where(mask.bool(), minus_inf, score)
        return score
