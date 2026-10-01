import types

import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as _NmsPy

# small constant to keep probabilities away from 0/1 (avoids inf in the prob<->llr maps)
EPS = 1e-12


def _loo_prod(t):
    """Leave-one-out product along dim=2 via exclusive prefix * suffix products."""
    ones = torch.ones_like(t[:, :, :1])
    prefix = torch.cumprod(t, dim=2)
    prefix_excl = torch.cat([ones, prefix[:, :, :-1]], dim=2)
    suffix = torch.flip(torch.cumprod(torch.flip(t, dims=[2]), dim=2), dims=[2])
    suffix_excl = torch.cat([suffix[:, :, 1:], ones], dim=2)
    return prefix_excl * suffix_excl


def _cn_q(a, syndrome_odd):
    """Check-node probability Q = (1 - (1 - 2 s_j) * prod_{k!=i} t_k) / 2 on the
    [batch, n_checks, degree] v->c LLRs `a`, with t = 1 - 2 sigmoid(-a), both
    probabilities clamped to [EPS, 1 - EPS]. Overwrites `a`."""
    t = a.neg_().sigmoid_().clamp_(EPS, 1.0 - EPS).mul_(-2.0).add_(1.0)
    t_excl = _loo_prod(t)
    q = torch.where(syndrome_odd, t_excl, -t_excl)
    return q.add_(1.0).div_(2.0).clamp_(EPS, 1.0 - EPS)


def _step(l_v, c2v_prev, u_init, syndrome_odd, beta, col, vn_adj, mask_dummy):
    """One decoding iteration for the compiled path, with the arithmetic of the
    eager methods: v2c, sum-product check-node update, c2v sum onto the channel
    LLR in (c, k) order, hard decision and syndrome. `beta` is unused (the
    signature is bp_norm_min_sum's). Returns new (c2v_msg, l_v, e_v, s_est)."""
    B, N1 = l_v.shape
    M, D = mask_dummy.shape
    a = l_v[:, col].view(B, M, D) - c2v_prev
    q = _cn_q(a, syndrome_odd)
    msg = torch.log((1.0 - q) / q).masked_fill(mask_dummy, 0.0)
    flat = torch.cat([msg.view(B, -1), msg.new_zeros(B, 1)], 1)
    g = flat[:, vn_adj].view(B, -1, N1)
    s = u_init
    for k in range(g.shape[1]):
        s = s + g[:, k]
    e = s <= 0.0
    s_est = e.to(torch.uint8)[:, col].view(B, M, D).sum(dim=2, dtype=torch.uint8) & 1
    return msg, s, e.to(l_v.dtype), s_est.to(l_v.dtype)


class create(_NmsPy):
    """
    This class creates a probability-domain sum-product (SPA) bp decoder.

    This is the deterministic reference oracle for the stochastic-computing decoder
    `bp_sum_prod_sc`: it computes the exact probability-domain message-passing equations
    that the stochastic XOR/equality logic gates approximate. The only difference from
    `bp_norm_min_sum` is the check-node update, which uses the syndrome-folded
    probability-domain XOR rule (sum-product / tanh rule) instead of normalized min-sum.
    The loop, row compaction, compiled step (CUDA) and hooks are bp_norm_min_sum's; the
    posterior LLR is the channel LLR plus the c->v messages added one at a time in
    (check, slot) order.

    References:
    R. G. Gallager, Low-Density Parity-Check Codes, MIT Press, 1963. doi:10.7551/mitpress/4347.001.0001
    F. R. Kschischang, B. J. Frey, H.-A. Loeliger, "Factor graphs and the sum-product algorithm," IEEE Trans. Inf. Theory, vol. 47, no. 2, pp. 498-519, 2001. doi:10.1109/18.910572
    D. Poulin, Y. Chung, "On the iterative decoding of sparse quantum codes," Quantum Inf. Comput., vol. 8, no. 10, pp. 987-1000, 2008. arXiv:0801.1241
    """

    def __init__(self, decoder_cfg, **kwargs) -> None:
        """Takes every bp_norm_min_sum key (max_iter, dtype, check_type, compile,
        rebatch_opt, rebatch_opt_params)."""
        super().__init__(decoder_cfg, **kwargs)
        self.algo = "bp_sum_prod"
        if self.compile:
            step = types.FunctionType(
                _step.__code__.replace(), _step.__globals__, _step.__name__
            )
            self._step = torch.compile(step)
        logger.info("bp_sum_prod (probability-domain sum-product) ready.")

    def cn_update(self, a_v2c, syndrome_odd, out):
        """Probability-domain sum-product (XOR) check-node update with syndrome folding.

        For check j and incident variable i:
            Q_{j->i} = (1 - (1 - 2 s_j) * prod_{k!=i} (1 - 2 p_{k->j})) / 2
        where p = P(bit = 1) = sigmoid(-LLR). This is the exact tanh rule the stochastic
        XOR gate approximates; min-sum replaces the product by its dominant term.
        The result is the c->v LLR log((1 - Q) / Q), dummy slots 0. Overwrites
        `a_v2c`. Writes into and returns `out`.
        """
        q = _cn_q(a_v2c, syndrome_odd)
        torch.div(torch.rsub(q, 1.0), q, out=out).log_()
        out.view(self.batch_size, -1).index_fill_(1, self.dummy_idx, 0.0)
        return out

    def c2v(self, c2v_flat, out, gathered):
        """Gathers the c->v messages of each variable through `self.vn_adj` into
        `gathered` and returns it as [batch, VD, N+1]; llr_update sums them."""
        torch.gather(c2v_flat, 1, self.vn_adj.expand(self.batch_size, -1), out=gathered)
        return gathered.view(self.batch_size, -1, out.shape[1])

    def llr_update(self, u_init, b_c2v, out):
        """Posterior LLR: the channel LLR, then each variable's c->v messages
        ([batch, VD, N+1] from c2v) added one at a time in (c, k) order, padding
        adding 0. The dummy variable (last column) stays +inf. Writes into and
        returns `out`."""
        out.copy_(u_init)
        for slot in b_c2v.unbind(1):
            out.add_(slot)
        return out
