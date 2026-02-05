from __future__ import annotations

import torch

class FPSG(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        a_min: torch.Tensor,
        a_max: torch.Tensor,
        beta: float,
        leak: float,
    ) -> torch.Tensor:
        a_dtype = a.dtype
        a = a.to(torch.float32)
        lo = torch.minimum(a_min, a_max).to(torch.float32).clamp_min(0.0)
        hi = torch.maximum(a_min, a_max).to(torch.float32).clamp_min(0.0)

        lam_lo = (a - hi).amin(dim=1, keepdim=True)
        lam_hi = (a - lo).amax(dim=1, keepdim=True)

        scale = a.abs().mean(dim=1, keepdim=True).clamp_min(1.0)
        eps = 1e-2 * scale
        lam_lo = lam_lo - eps
        lam_hi = lam_hi + eps

        for _ in range(60):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            a_mid = (a - lam_mid).clamp_min(lo).clamp_max(hi)
            s = a_mid.sum(dim=1, keepdim=True)
            go_right = (s > 1.0)
            lam_lo = torch.where(go_right, lam_mid, lam_lo)
            lam_hi = torch.where(go_right, lam_hi, lam_mid)

        lam = 0.5 * (lam_lo + lam_hi)
        w = (a - lam).clamp_min(lo).clamp_max(hi)

        ctx.save_for_backward(a, lo, hi, lam)
        ctx.beta = float(beta)
        ctx.leak = float(leak)
        ctx.a_dtype = a_dtype

        return w.to(a_dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, lo, hi, lam = ctx.saved_tensors
        beta = ctx.beta
        leak = ctx.leak

        go = grad_out.to(a.dtype)
        x = a - lam
        indicator = ((x > lo) & (x < hi)).to(a.dtype)
        smooth_gate = torch.sigmoid(beta * (x - lo)) - torch.sigmoid(beta * (x - hi))
        g = indicator + (1.0 - indicator) * (leak + (1.0 - leak) * smooth_gate)

        G = g.sum(dim=1, keepdim=True).clamp_min(1e-12)
        v = go * g
        c = v.sum(dim=1, keepdim=True) / G
        grad_a = v - g * c

        return (
            grad_a.to(ctx.a_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )