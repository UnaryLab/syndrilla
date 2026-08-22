"""Logical-centric loss for the saq decoder (SAQ paper, Sec. "Logical-Centric Loss").

Supervises the logical class rather than the error bit pattern, so degenerate solutions
differing by a stabilizer are not penalised. Reads the `llr`, `logical_logits` and
`logical_prior` entries the decoder writes into its io_dict, so it is coupled to that
output contract rather than to any one decoder's internals.

Not an nn.Module: it has no learnable parameters, and assigning the decoder to an
nn.Module attribute would register the whole decoder as a submodule of the loss, putting
the decoder's parameters into `loss.parameters()` and `loss.state_dict()`.
"""

import torch
import torch.nn.functional as F


# Upstream's Codes.py GF(2) <-> +-1 conversions, kept here rather than imported from the
# decoder: the loss stands alone, and the repo inlines its GF(2) algebra per module.
def bin_to_sign(x):
    return 1 - 2 * x


def sign_to_bin(x):
    return 0.5 * (1 - x)


def _bits_to_index(bits):
    """Pack a [B, k] bit matrix into a [B] class index (LSB = bits[:, 0])."""
    weights = 2 ** torch.arange(bits.size(1), device=bits.device)
    return (bits * weights).sum(dim=1).long()


def _logical_flipped(L, x):
    """GF(2) product x @ L, used to map an error vector to its logical syndrome."""
    return torch.matmul(x.to(L.dtype), L) % 2


def _diff_GF2_mul(H, x):
    """Differentiable GF(2) matrix-vector product used by the entropy loss.

    Replaces the XOR over each column's support with a product of +-1 Bernoulli means, so
    gradients flow from the logical-error objective back to the per-qubit logits.
    """
    tmp = bin_to_sign(H.unsqueeze(0) * x.unsqueeze(-1))
    return sign_to_bin(torch.prod(tmp, 1))


class create:
    """The three-term logical-centric objective, bound to the decoder it supervises."""

    def __init__(self, loss_cfg, **kwargs) -> None:
        decoder = kwargs.get("decoder")
        if decoder is None:
            raise ValueError(
                "Loss <logical_centric> requires the decoder it supervises, passed as the "
                "<decoder> kwarg of create_loss()."
            )
        # the RoundFlattenWrapper forwards unknown attributes, but bind to the inner
        # module so `logic_matrix`, `device` and `dtype` resolve directly
        self.decoder = getattr(decoder, "decoder", decoder)
        self.lambda_lc = float(loss_cfg.get("lambda_lc", 1.0))
        self.lambda_lp = float(loss_cfg.get("lambda_lp", 0.2))
        self.lambda_ent = float(loss_cfg.get("lambda_ent", 1.0))

    def _prepare(self, e):
        """Cast the ground-truth error to the decoder's device/dtype and pack its class."""
        e = e.to(device=self.decoder.device, dtype=self.decoder.dtype)
        target_idx = _bits_to_index(
            _logical_flipped(self.decoder.logic_matrix, e).long()
        )
        return e, target_idx

    def terms(self, io_dict, e):
        """The three unweighted terms (L_LC, L_LP, L_Ent).

        Input:
            io_dict: the decoder's output dict, needing 'llr', 'logical_logits' and
                'logical_prior'
            e: ground-truth error, [batch, n] in {0, 1}

            L_LC   cross-entropy on the transformer's logical class output
            L_LP   cross-entropy on the shallow prior, same target
            L_Ent  differentiable GF(2) logical entropy: pushes the *logical* syndrome of
                   the residual `e XOR prediction` towards zero rather than matching the
                   error bitwise, so degenerate solutions are not penalised

        Use `combine` to weight them with the configured lambdas.

        Note on L_Ent: `_diff_GF2_mul` expects per-qubit `P(bit == 1)`, so the residual must
        be supplied as `P(e_i XOR pred_i == 1)`. Upstream passes its complement
        (`z*(1-sigmoid) + (1-z)*sigmoid`, i.e. `P(residual == 0)`). Since
        `XOR_i not(r_i) == (XOR_i r_i) XOR (w mod 2)` over a logical operator of weight `w`,
        that complement leaves the term correct for even-weight logicals but *inverts* it for
        odd-weight ones, where minimising it maximises the logical error rate. Measured on a
        perfect prediction vs one off by a logical operator: rotated surface d=5 (weight 5)
        gives 6.39 vs 0.0017 upstream, exactly backwards; toric L=10 (weight 10) gives
        0.0034 vs 2.85 either way. The correct form below is identical on even weights and
        fixes the odd ones.
        """
        e, target_idx = self._prepare(e)

        loss_lc = F.cross_entropy(io_dict["logical_logits"], target_idx)
        loss_lp = F.cross_entropy(io_dict["logical_prior"], target_idx)

        # positive LLR means "no error", so sigmoid(l_v) is P(prediction == 0)
        p_no_err = torch.sigmoid(io_dict["llr"])
        p_residual = e * p_no_err + (1 - e) * (1 - p_no_err)
        logical_residual = _diff_GF2_mul(self.decoder.logic_matrix, p_residual)
        loss_ent = F.binary_cross_entropy(
            logical_residual, torch.zeros_like(logical_residual)
        )

        return loss_lc, loss_lp, loss_ent

    def combine(self, loss_lc, loss_lp, loss_ent):
        """The three terms weighted by the configured lambdas, as one scalar.

        Split from __call__ so a training loop that also logs the individual terms
        computes them once instead of twice.
        """
        return (
            self.lambda_lc * loss_lc
            + self.lambda_lp * loss_lp
            + self.lambda_ent * loss_ent
        )

    def __call__(self, io_dict, e):
        """`combine(*terms(...))`, for callers that only want the scalar."""
        return self.combine(*self.terms(io_dict, e))

    def class_error(self, io_dict, e):
        """Fraction of the batch whose predicted logical class is wrong.

        Uses the same target packing `terms` trains against, which is why it lives here
        rather than in metric/.
        """
        _, target_idx = self._prepare(e)
        predicted = io_dict["logical_logits"].argmax(dim=1)
        return (predicted != target_idx).float().mean().item()
