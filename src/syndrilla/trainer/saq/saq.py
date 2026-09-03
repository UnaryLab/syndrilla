import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


def _require_number(value, key):
    """Return a required numeric optimizer setting as a float."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"Trainer <saq> requires a numeric <{key}> under the training yaml's "
            f"`training.optimizer`, got <{value!r}>."
        )
    return float(value)


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
    """Differentiable GF(2) matrix-vector product, in the probability domain."""
    tmp = bin_to_sign(H.unsqueeze(0) * x.unsqueeze(-1))
    return sign_to_bin(torch.prod(tmp, 1))


def _parity_llr(H, llr):
    """LLR of the GF(2) parity of `llr`'s bits over each column's support."""
    mask = H.t().unsqueeze(0).to(llr.dtype)  # [1, k, n]
    x = llr.unsqueeze(1)  # [B, 1, n]

    # sign: exact, and a step function of the llrs, so it carries no gradient of its own
    negatives = (x < 0).to(llr.dtype) * mask
    sign = bin_to_sign(negatives.sum(dim=-1) % 2)  # [B, k]

    outside = llr.new_tensor(float("inf"))
    magnitude = torch.where(mask.bool(), x.abs(), outside)  # [B, k, n]
    return sign * magnitude.min(dim=-1).values


class create:
    """How saq is trained: its three-term objective and the optimizer that descends it."""

    term_names = ("lc", "lp", "ent")

    def __init__(self, cfg, **kwargs) -> None:
        decoder = kwargs.get("decoder")
        if decoder is None:
            raise ValueError(
                "Trainer <saq> requires the decoder it supervises, passed as the "
                "<decoder> kwarg of create_trainer()."
            )
        self.decoder = getattr(decoder, "decoder", decoder)
        # the whole `training` block arrives, the way a decoder is handed its own; the
        # term weights are the part of it the objective reads
        loss_cfg = cfg.get("loss") or {}
        self.cfg = dict(loss_cfg)
        self.lambda_lc = float(loss_cfg.get("lambda_lc", 1.0))
        self.lambda_lp = float(loss_cfg.get("lambda_lp", 0.2))
        self.lambda_ent = float(loss_cfg.get("lambda_ent", 1.0))

    def configure_optimizer(self, optimizer_cfg, parameters, epochs):
        """Adam plus a cosine schedule over `epochs`, from `training.optimizer`."""
        if not isinstance(optimizer_cfg, dict) or not optimizer_cfg:
            raise ValueError(
                f"Trainer <saq>: <training.optimizer> is a block with <lr>, "
                f"<weight_decay> and <min_lr>, got <{optimizer_cfg!r}>."
            )
        lr = _require_number(optimizer_cfg.get("lr"), "lr")
        weight_decay = _require_number(
            optimizer_cfg.get("weight_decay"), "weight_decay"
        )
        min_lr = _require_number(optimizer_cfg.get("min_lr"), "min_lr")

        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        # nothing has run backward yet, so the first step must not find stale gradients
        optimizer.zero_grad(set_to_none=True)
        return optimizer, scheduler

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
        """The three terms weighted by the configured lambdas, as one scalar."""
        return (
            self.lambda_lc * loss_lc
            + self.lambda_lp * loss_lp
            + self.lambda_ent * loss_ent
        )

    def __call__(self, io_dict, e):
        """`combine(*terms(...))`, for callers that only want the scalar."""
        return self.combine(*self.terms(io_dict, e))

    def class_error(self, io_dict, e):
        """Fraction of the batch whose predicted logical class is wrong."""
        _, target_idx = self._prepare(e)
        predicted = io_dict["logical_logits"].argmax(dim=1)
        return (predicted != target_idx).float().mean().item()
