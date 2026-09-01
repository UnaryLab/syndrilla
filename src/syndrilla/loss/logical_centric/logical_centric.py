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
    """Differentiable GF(2) matrix-vector product, in the probability domain.

    The XOR over each column's support becomes a product of +-1 Bernoulli means: with
    `x_i = P(bit_i == 1)`, each factor `1 - 2*x_i` is `E[(-1)^bit_i]`, and their product
    is `E[(-1)^parity]`, so `sign_to_bin` of it is `P(parity == 1)`.

    Exact, and unusable as written on a support of any size: every factor has magnitude
    below 1, so the product decays geometrically in the number of bits and takes its own
    gradient down with it. `_parity_llr` computes the same quantity in the log domain
    and is what the loss uses; this is kept because it is the plainest statement of what
    that one approximates, and the tests check the two against each other.
    """
    tmp = bin_to_sign(H.unsqueeze(0) * x.unsqueeze(-1))
    return sign_to_bin(torch.prod(tmp, 1))


def _parity_llr(H, llr):
    """LLR of the GF(2) parity of `llr`'s bits over each column's support.
    """
    mask = H.t().unsqueeze(0).to(llr.dtype)  # [1, k, n]
    x = llr.unsqueeze(1)  # [B, 1, n]

    # sign: exact, and a step function of the llrs, so it carries no gradient of its own
    negatives = (x < 0).to(llr.dtype) * mask
    sign = bin_to_sign(negatives.sum(dim=-1) % 2)  # [B, k]

    outside = llr.new_tensor(float("inf"))
    magnitude = torch.where(mask.bool(), x.abs(), outside)  # [B, k, n]
    return sign * magnitude.min(dim=-1).values


class create:
    """The three-term logical-centric objective, bound to the decoder it supervises."""
    term_names = ("lc", "lp", "ent")

    def __init__(self, loss_cfg, **kwargs) -> None:
        decoder = kwargs.get("decoder")
        if decoder is None:
            raise ValueError(
                "Loss <logical_centric> requires the decoder it supervises, passed as the "
                "<decoder> kwarg of create_loss()."
            )
        self.decoder = getattr(decoder, "decoder", decoder)
        self.cfg = dict(loss_cfg)
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
        """
        e, target_idx = self._prepare(e)

        loss_lc = F.cross_entropy(io_dict["logical_logits"], target_idx)
        loss_lp = F.cross_entropy(io_dict["logical_prior"], target_idx)

        # positive llr means "no error", so the residual's llr is the decoder's own with
        # the sign flipped on the bits the true error flips
        llr_residual = bin_to_sign(e) * io_dict["llr"]
        parity = _parity_llr(self.decoder.logic_matrix, llr_residual)
        loss_ent = F.softplus(-parity).mean()

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
