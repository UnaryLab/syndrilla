"""Sum-product BP with oscillating-trapping-set prior switches.

Dimitris Chytas, Michele Pacenti, Nithin Raveendran, Mark F. Flanagan and
Bane Vasic,
"Enhanced Message-Passing Decoding of Degenerate Quantum Codes Utilizing
Trapping Set Dynamics," IEEE Communications Letters, 2024, Algorithm 1.
https://doi.org/10.1109/LCOMM.2024.3356312
Author manuscript: https://par.nsf.gov/servlets/purl/10493558

The paper uses sum-product updates, T=9 for surface/toric codes and T=7 for
bicycle codes; it specifies no numeric C. Here interval=10 and strength=10
retain the existing configuration defaults. An empty oscillator set selects
only the least-reliable node, resolving the pseudocode's undefined j1.
Equal oscillator counts select the smallest absolute posterior, then the lowest
node index. The least-reliable pick also breaks ties by node index. If both
picks select the same node, apply one pin.
The shared decoder convention maps a zero posterior to hard bit one.
"""

import math

import torch

from syndrilla.decoder.bp_sum_prod.bp_sum_prod import create as _SumProduct
from syndrilla.decoder.knobs import reject_unknown_keys


def _validate(cfg):
    interval = cfg.get("interval", 10)
    strength = cfg.get("strength", 10.0)
    if type(interval) is not int or not 0 < interval < 2**31:
        raise ValueError("interval must be a positive int below 2^31")
    if (type(strength) not in (int, float) or not math.isfinite(strength)
        or strength <= 0):
        raise ValueError("strength must be finite and positive")
    maximum = cfg.get("max_iter", 50)
    if type(maximum) is not int or not 0 < maximum < 2**31:
        raise ValueError("max_iter must be a positive int below 2^31")
    if cfg.get("dtype", "float64") not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if cfg.get("check_type", "hx") not in ("hx", "hz"):
        raise ValueError("check_type must be hx or hz")
    return interval, strength


class create(_SumProduct):
    """Keep BP messages while periodically replacing two channel priors."""

    _config_keys = ("interval", "strength")
    _config_algorithm = "bp_ots"

    def __init__(self, cfg, **kwargs):
        reject_unknown_keys(
            cfg, self._base_config_keys.union(self._config_keys),
            cfg.get("algorithm", self._config_algorithm),
        )
        self.interval, self.strength = _validate(cfg)
        super().__init__(cfg, **kwargs)
        self.algo = "bp_ots"

    def forward(self, io_dict):
        base = io_dict["llr0"].to(self.device, self.dtype)
        if base.ndim != 2 or base.shape[1] != self.H_shape[1]:
            raise ValueError("llr0 must have shape [batch, variables]")
        if io_dict["synd"].shape != (len(base), self.H_shape[0]):
            raise ValueError("synd must match the batch and check dimensions")
        self.cap_active_last = False
        self._ots_channel = base.clone()
        self._ots_counts = torch.zeros_like(base, dtype=torch.int64)
        self._ots_previous = base <= 0
        self._ots_current = base.clone()
        self._ots_prior = None
        self.last_ots_nodes = None
        if len(base) == 0:
            io_dict.update(llr=base.clone(), e_v=base.clone(),
                iter=torch.empty(0, device=self.device, dtype=torch.long),
                converge=torch.empty(0, device=self.device, dtype=torch.long),
                defer=torch.empty(0, device=self.device, dtype=torch.bool))
            return io_dict
        step = getattr(self, "_step", None)
        try:
            if self.compile:
                def ots_step(l_v, messages, prior, *args):
                    self._ots_prior = prior
                    result = step(l_v, messages, prior, *args)
                    if self.i < self.max_iter and self.i % self.interval == 0:
                        incoming = result[0].reshape(len(l_v), -1)
                        flat = torch.cat((incoming, incoming.new_zeros((len(l_v), 1))), 1)
                        self._ots_incoming = flat[:, self.vn_adj].view(len(l_v), -1, l_v.shape[1])
                    return result
                self._step = ots_step
            return super().forward(io_dict)
        finally:
            if step is not None:
                self._step = step
            self._ots_prior = None
            self._ots_incoming = None

    def llr_update(self, prior, messages, out):
        self._ots_prior = prior
        self._ots_incoming = messages
        return super().llr_update(prior, messages, out)

    def _iter_hook(self, i, l_v, e_v, active, syndrome):
        rows = self._hook_rows
        hard = e_v[:, :-1].bool()
        self._ots_counts[rows] += (hard != self._ots_previous[rows]).long()
        self._ots_previous[rows] = hard
        if i >= self.max_iter or i % self.interval:
            return
        self._switch_priors(rows, l_v, active)

    def _switch_priors(self, rows, l_v, active):
        magnitude = l_v[:, :-1].abs()
        counts = self._ots_counts[rows]
        maximum = counts.max(1, keepdim=True).values
        tied = (counts == maximum) & (maximum > 0)
        oscillator = magnitude.masked_fill(~tied, float("inf")).argmin(1)
        has_oscillator = active & (maximum[:, 0] > 0)
        least = magnitude.argmin(1)
        prior = self._ots_channel[rows].clone()
        local = active.nonzero().flatten()
        prior[local, least[local]] = -self.strength
        oscillating = has_oscillator.nonzero().flatten()
        prior[oscillating, oscillator[oscillating]] = -self.strength
        self._ots_counts[rows[oscillating], oscillator[oscillating]] = 0
        # Recompute the outgoing posterior from the changed prior and retained messages.
        changed = torch.where(active[:, None], prior, self._ots_current[rows])
        self._ots_prior[:, :-1].copy_(changed)
        _SumProduct.llr_update(self, self._ots_prior, self._ots_incoming, l_v)
        self._ots_current[rows] = changed
        self.last_ots_nodes = torch.stack((torch.where(has_oscillator, oscillator, -1),
                                          torch.where(active, least, -1)), 1)
