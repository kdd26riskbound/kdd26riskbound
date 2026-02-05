import torch, torch.nn as nn
from torch.optim          import Adam
from torch.distributions  import Normal
from typing               import Optional

from .model.ASU import ASU
from .model.MSU import MSU
EPS = 1e-12

def _to_tensor(arr, device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    arr = torch.from_numpy(arr)
    if not torch.is_floating_point(arr):
        arr = arr.to(dtype)
    return arr.to(device)

class RLActor(nn.Module):
    def __init__(self, supports, args):
        super().__init__()
        self.args = args

        self.asu = ASU(args.num_assets,
                       args.in_features[0],
                       args.hidden_dim,
                       args.window_len,
                       args.dropout,
                       args.kernel_size,
                       args.num_blocks,
                       supports,
                       args.spatial_bool,
                       args.addaptiveadj)

        self.msu = (MSU(args.in_features[1], args.window_len,
                        args.hidden_dim) if args.msu_bool else None)


    def forward(self,
                xa: torch.Tensor,
                xm: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None,
                deterministic: bool = False):

        scores = self.asu(xa, mask)

        rho, rho_log = None, None
        if self.msu is not None and xm is not None:
            raw = self.msu(xm)

            if isinstance(raw, (list, tuple)):
                mu, log_sigma = raw[0], raw[1]
            elif isinstance(raw, torch.Tensor):
                if raw.dim() < 2 or raw.size(-1) < 2:
                    raise ValueError("last dimension of MSU output must be at least 2.")
                mu, log_sigma = raw[:, 0], raw[:, 1]
            else:
                raise TypeError("Unsupported type for MSU output.")

            sigma = torch.log1p(torch.exp(log_sigma))
            dist  = Normal(mu, sigma)

            sample   = dist.rsample()
            rho      = mu.clamp(0., 1.) if deterministic else sample.clamp(0., 1.)
            rho_logt = None if deterministic else dist.log_prob(sample)
            rho_log  = rho_logt
        else:
            rho     = torch.full((scores.size(0),), 0.5, device=scores.device)
            rho_log = None
        B, N = scores.shape
        G    = self.args.G
        weights = torch.zeros(B, 2*N, device=scores.device)


        long_val, long_idx = torch.topk(scores,    G, dim=-1)
        long_w = torch.softmax(long_val, dim=-1)
        weights.scatter_(1, long_idx, long_w)


        loser_score          = scores.sign() * (1.0 - scores)
        short_val, short_idx = torch.topk(loser_score, G, dim=-1)
        short_w              = torch.softmax(short_val, dim=-1)
        weights.scatter_(1, short_idx + N, short_w)

        return weights, rho, torch.softmax(scores, -1), rho_log

class RLAgent:
    def __init__(self, env, actor, args):
        self.env   = env
        self.actor = actor
        self.args  = args
        self.opt   = Adam(self.actor.parameters(), lr=args.lr)

    def train_episode(self):
        self.actor.train()
        states, mask = self.env.reset()
        xa, xm       = states
        xa   = _to_tensor(xa, self.args.device)
        xm   = _to_tensor(xm, self.args.device) if xm is not None else None
        mask = _to_tensor(mask, self.args.device, torch.bool)

        weights, rho, p_a, _ = self.actor(xa, xm, mask)

        loss_acc, steps = 0.0, 0
        done = False
        while not done:
            (xa_n, xm_n), rewards, _, mask_n, done, info = \
                self.env.step(weights, rho)

            reward = torch.from_numpy(info['reward'].total) \
                         .to(self.args.device)
            loss = -(reward *
                     torch.log(p_a.gather(1,
                             torch.argmax(p_a, 1, keepdim=True)) + EPS)).mean()

            if torch.isnan(loss): break
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt.step()

            loss_acc += loss.item();  steps += 1
            if done:
                break

            xa   = _to_tensor(xa_n, self.args.device)
            xm   = _to_tensor(xm_n, self.args.device) if xm_n is not None else None
            mask = _to_tensor(mask_n, self.args.device, torch.bool)

            weights, rho, p_a, _ = self.actor(xa, xm, mask)

        return loss_acc / max(steps, 1)

    @torch.no_grad()
    def infer(self):
        self.actor.eval()
        self.env.set_test()
        states, mask = self.env.reset()
        xa, xm       = states

        xa = _to_tensor(xa, self.args.device)
        xm = _to_tensor(xm, self.args.device) if xm is not None else None
        mask = _to_tensor(mask, self.args.device, torch.bool)

        weights, rho, _, _ = self.actor(xa, xm, mask, deterministic=True)
        return weights.cpu().numpy(), rho.cpu().numpy()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.args.device, weights_only=True))