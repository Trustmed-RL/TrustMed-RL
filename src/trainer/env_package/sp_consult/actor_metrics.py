"""Actor-side instrumentation the fork calls, kept HERE so it can be unit-tested without a GPU box."""

from __future__ import annotations

import torch


def _f(t) -> float:
    return float(t.detach().item()) if torch.is_tensor(t) else float(t)


def _frac(num: torch.Tensor, den: torch.Tensor) -> float:
    """num and den are boolean masks; 0.0 when the denominator is empty (the count keys say so)."""
    d = den.sum()
    if d.item() == 0:
        return 0.0
    return float((num.sum().float() / d.float()).item())


def success_clip_metrics(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_high: float,
    clip_low: float,
    row_select: torch.Tensor | None = None,
    prefix: str = "actor/",
) -> dict:
    """PPO clipping, split by SIGN and by whether the row's trajectory got the diagnosis right."""
    with torch.no_grad():
        m = response_mask.bool()
        ratio = torch.exp(log_prob.detach() - old_log_prob.detach())
        pos = m & (advantages > 0)
        neg = m & (advantages < 0)
        clip_hi = pos & (ratio > 1.0 + float(clip_high))
        clip_lo = neg & (ratio < 1.0 - float(clip_low))
        out = {
            f"{prefix}clipfrac_pos": _frac(clip_hi, pos),
            f"{prefix}clipfrac_neg": _frac(clip_lo, neg),
            f"{prefix}tok_pos": _f(pos.sum()),
            f"{prefix}tok_neg": _f(neg.sum()),
            f"{prefix}tok_clip_pos": _f(clip_hi.sum()),
            f"{prefix}tok_clip_neg": _f(clip_lo.sum()),
            f"{prefix}tok_total": _f(m.sum()),
        }
        if row_select is None:
            return out
        sel = row_select.detach().bool().reshape(-1, 1).expand_as(m)
        dx_tok = m & sel
        dx_pos = pos & sel
        dx_clip = clip_hi & sel
        out.update(
            {
                f"{prefix}clipfrac_pos_dx": _frac(dx_clip, dx_pos),
                f"{prefix}tok_pos_dx": _f(dx_pos.sum()),
                f"{prefix}tok_clip_pos_dx": _f(dx_clip.sum()),
                f"{prefix}tok_dx": _f(dx_tok.sum()),
                f"{prefix}dx_token_share": _frac(dx_tok, m),
                f"{prefix}ratio_mean_dx_pos": (
                    float(ratio[dx_pos].mean().item()) if dx_pos.any() else 0.0
                ),
                f"{prefix}ratio_p90_dx_pos": (
                    float(torch.quantile(ratio[dx_pos].float(), 0.9).item())
                    if dx_pos.any()
                    else 0.0
                ),
                f"{prefix}adv_mean_dx_pos": (
                    float(advantages[dx_pos].mean().item()) if dx_pos.any() else 0.0
                ),
            }
        )
        return out


def clip_metrics(preclip_norm, grad_clip: float, prefix: str = "actor/") -> dict:
    """What global gradient clipping actually did this step."""
    p = _f(preclip_norm)
    c = float(grad_clip)
    scale = 1.0 if p <= c or p == 0.0 else c / p
    return {
        f"{prefix}grad_norm_preclip": p,
        f"{prefix}grad_norm_postclip": p * scale,
        f"{prefix}grad_clip_scale": scale,
        f"{prefix}grad_clipped": 1.0 if scale < 1.0 else 0.0,
    }


class ParamDeltaProbe:
    """||theta_after - theta_before|| of one optimizer step, on a fixed deterministic sample."""

    def __init__(self, params, max_elems: int = 1 << 21, prefix: str = "actor/"):
        self.prefix = prefix
        self._slices: list[tuple[torch.nn.Parameter, int, int]] = []
        self._before: list[torch.Tensor] = []
        budget = int(max_elems)
        for p in params:
            if not torch.is_tensor(p) or p.numel() == 0 or budget <= 0:
                continue
            n = min(p.numel(), budget)
            self._slices.append((p, 0, n))
            budget -= n
        self.n_elems = sum(n for _, _, n in self._slices)

    def snapshot(self) -> None:
        self._before = [p.detach().reshape(-1)[a:b].clone().float() for p, a, b in self._slices]

    def delta(self) -> dict:
        if not self._before:
            return {}
        with torch.no_grad():
            du = torch.zeros((), dtype=torch.float64)
            pn = torch.zeros((), dtype=torch.float64)
            for (p, a, b), old in zip(self._slices, self._before):
                new = p.detach().reshape(-1)[a:b].float()
                du += (new - old).pow(2).sum().double().cpu()
                pn += new.pow(2).sum().double().cpu()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                buf = (
                    torch.stack([du, pn]).cuda()
                    if torch.cuda.is_available()
                    else torch.stack([du, pn])
                )
                torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
                du, pn = buf[0].cpu(), buf[1].cpu()
            u, q = float(du.sqrt().item()), float(pn.sqrt().item())
        self._before = []
        return {
            f"{self.prefix}update_norm": u,
            f"{self.prefix}param_norm": q,
            f"{self.prefix}update_ratio": (u / q if q > 0 else 0.0),
            f"{self.prefix}update_probe_elems": float(self.n_elems),
        }
