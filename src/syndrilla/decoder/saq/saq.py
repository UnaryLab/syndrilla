"""SAQ -- Stabilizer-Aware Quantum error correction decoder.

PyTorch port of https://github.com/DavidZenati/SAQ-Decoder (Zenati & Nachmani,
"SAQ: Stabilizer-Aware Quantum Error Correction Decoder", ICLR 2026, arXiv:2512.08914),
adapted to the syndrilla decoder contract.

Unlike the belief-propagation decoders in this package, SAQ is a *learned* decoder: it
maps a syndrome to an error estimate in a single feed-forward pass, so there is no
message-passing loop and no per-sample iteration count. The decoding pipeline mirrors
the paper's stages:

  Stage 1  Dual-stream representation construction
           `logical_prior` runs a shallow MLP over the syndrome to get initial logical
           class logits; `build_streams` turns the syndrome and those logits into two
           token streams (syndrome stream SN, prefixed with a global token, and logical
           stream LN).
  Stage 2  Syndrome-Logical Transformer Decoder (SLTD)
           `layer_update` applies one SLTD layer: `sn_update` runs topology-masked self-attention
           over the syndrome stream, then `ln_update` runs the reverse (S -> L) cross-attention
           that lets the logical tokens read the freshly updated syndrome tokens.
           `head_update` projects the final token states back to a per-qubit LLR vector and
           logical class logits.
  Stage 3  Constraint-Projected Nullspace Descent (CPND), inference only
           `project` maps the hard decision onto an operator that satisfies the measured
           syndrome and the predicted logical class exactly over GF(2); `nullspace_descent`
           then walks the stabilizer coset that preserves both, greedily lightening the
           operator under the posterior LLR weights. Enabled by default, `cpnd: false`
           disables it.

Samples whose estimate does not reproduce the measured syndrome are reported as unconverged,
so a chained decoder (e.g. `osd_0`) can retry them. With CPND on, that can only happen when
the measured syndrome is not in the image of H, which is possible under noisy syndrome
extraction but not under perfect measurement.

`configure_optimizer`, `backward` and `update` are the training stages: the decoder owns its
optimizer and applies its own gradient. The objective itself lives in
`syndrilla/loss/logical_centric/`, which reads the `llr`, `logical_logits` and
`logical_prior` entries `forward` writes. Trained weights come back through the
`checkpoint` config key.

Supported codes: toric and surface, rotated or not. The family is measured from the
matrix, not configured.
"""

import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR

from syndrilla.utils import parse_device_dtype


# Upstream's Codes.py GF(2) <-> +-1 conversions. SAQ feeds the transformer a +-1 syndrome
# (bin_to_sign) while syndrilla passes syndromes and errors around as 0/1.
def bin_to_sign(x):
    return 1 - 2 * x


def _require_number(value, key):
    """Return a required numeric decoder setting as a float."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"saq requires a numeric <{key}> in the decoder config, got <{value!r}>. "
            f"Write exponent floats as 5.0e-4, not 5e-4: yaml reads the short form "
            f"as a string."
        )
    return float(value)


def _clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# --------------------------------------------------------------------------- #
# Stage 3: Constraint-Projected Nullspace Descent (CPND).
#
# Port of upstream's CPND.py, rewritten against numpy and torch so the `galois`
# dependency is not needed, and with two defects of the released implementation
# fixed (see `_cpnd_descent` and the weight discussion below).
#
# The transformer of stages 1 and 2 is an unconstrained predictor: nothing forces
# its output to satisfy `H e = s`. CPND repairs that in two steps, at inference only.
#
#   Projection        Stack the (row-independent) stabilizer checks and the logical
#                     operators into `H_hat = [H; L]` and precompute a right inverse
#                     `B` with `H_hat B = I`. Given a raw prediction and the predicted
#                     logical class, correcting by `B` times the constraint residual
#                     yields an operator satisfying `H e = s` and `L e = class` exactly.
#
#   Nullspace descent Everything satisfying those constraints lies in `e0 + ker(H_hat)`.
#                     Since the kernel is taken of `[H; L]` and not of `H` alone, every
#                     basis vector is a stabilizer: moving along it preserves both the
#                     syndrome and the logical class, so the descent only ever picks a
#                     lighter representative of the coset the transformer already chose.
#                     Basis vectors are walked greedily, accepting a move whenever it
#                     lowers the reliability-weighted operator weight.
#
# Reliability weights
#   `l_v` is the posterior LLR with the repo-wide convention "positive means the qubit
#   is more likely error-free", i.e. `l_v_i = log(P(e_i=0) / P(e_i=1))`. The negative
#   log-likelihood of a candidate operator is then, up to a constant, `sum_i e_i * l_v_i`,
#   so the per-qubit cost of setting a bit is `w_i = l_v_i` and the descent minimises
#   `sum_i e_i * w_i`.
#
#   Upstream instead computes `w = -log(p / (1 - p))` from `p = out_S.sigmoid()`, which is
#   the negation of the above -- the wrong direction for a weight-minimising descent. That
#   inversion is masked in the released code only because `out_S` has already been
#   overwritten with its hard decision by the time the weights are computed, which re-flips
#   the sign while collapsing every magnitude to `{0, 1}`. Both effects are corrected here
#   by weighting with `l_v` directly.
# --------------------------------------------------------------------------- #


def _rref_gf2(matrix):
    """Reduced row echelon form over GF(2).

    Returns `(R, T, pivots)` with `T @ matrix == R` (mod 2), `R` in *reduced* echelon form
    (eliminated above and below every pivot, which both `_right_inverse` and
    `_kernel_basis` rely on) and `T` invertible. `pivots.size` is the GF(2) rank.

    Eliminates a whole column of rows per pivot with numpy, unlike
    `syndrilla.utils.row_echelon`, whose per-element Python loop is too slow to run at
    decoder construction for the larger lattices.
    """
    a = (np.asarray(matrix) % 2).astype(np.uint8)
    rows, cols = a.shape
    r = a.copy()
    t = np.identity(rows, dtype=np.uint8)
    pivots = []
    row = 0
    for col in range(cols):
        if row == rows:
            break
        hit = np.flatnonzero(r[row:, col])
        if hit.size == 0:
            continue
        p = row + hit[0]
        if p != row:
            r[[row, p]] = r[[p, row]]
            t[[row, p]] = t[[p, row]]
        others = np.flatnonzero(r[:, col])
        others = others[others != row]
        if others.size:
            r[others] ^= r[row]
            t[others] ^= t[row]
        pivots.append(col)
        row += 1
    return r, t, np.asarray(pivots, dtype=np.int64)


def _independent_rows(matrix):
    """Indices of a maximal linearly independent subset of the rows, over GF(2).

    Column pivots of `matrix.T` are row pivots of `matrix`. Used to drop the redundant
    stabilizer rows a toric lattice carries (all checks of one type multiply to the
    identity) so that `[H; L]` can have full row rank.
    """
    return _rref_gf2(np.asarray(matrix).T)[2]


def _right_inverse(H_hat):
    """`B` of shape [n, r] with `H_hat @ B == I_r` over GF(2).

    With `T @ H_hat = R` in reduced echelon form and pivot columns `P`, `R[:, P] = I` gives
    `T = H_hat[:, P]^-1`, hence `H_hat[:, P] @ T = I` as well. Scattering `T` into the pivot
    rows of an otherwise zero matrix therefore yields a right inverse. One elimination
    replaces upstream's `r` separate `gf2_solve` calls.

    Raises ValueError when `H_hat` is not full row rank, rather than returning a matrix that
    silently fails the identity.
    """
    _, t, pivots = _rref_gf2(H_hat)
    rows, cols = np.shape(H_hat)
    if pivots.size != rows:
        raise ValueError(
            f"CPND needs [H; L] to have full row rank, but its rank is "
            f"<{pivots.size}> over <{rows}> rows. The logical operators must be "
            f"independent of the stabilizer rows."
        )
    B = np.zeros((cols, rows), dtype=np.uint8)
    B[pivots] = t
    return B


def _kernel_basis(H_hat):
    """Basis of `ker(H_hat)` as a [n, g] GF(2) matrix, one basis vector per column.

    Each free column `f` of the reduced echelon form gives a vector that is 1 at `f` and
    `R[i, f]` at pivot column `i`.

    These are the reduced-echelon generators, matching upstream. They are correct but not
    minimal weight: a basis of the code's actual plaquette/star stabilizers would give the
    descent shorter moves and so a better local optimum.
    """
    r_mat, _, pivots = _rref_gf2(H_hat)
    n = np.shape(H_hat)[1]
    free = np.setdiff1d(np.arange(n), pivots)
    basis = np.zeros((n, free.size), dtype=np.uint8)
    if free.size:
        basis[free, np.arange(free.size)] = 1
        basis[pivots] = r_mat[: pivots.size][:, free]
    return basis


def _logits_to_logical_bits(logits, k):
    """Logical class logits [B, 2^k] -> the class's bit pattern [B, k], LSB first.

    Inverse of the class packing the logical-centric loss trains against, so the
    class CPND pins is the same one the loss supervised.
    """
    index = logits.argmax(dim=1)
    shifts = torch.arange(k, device=index.device)
    return (index.unsqueeze(1) >> shifts) & 1


def _cpnd_project(e_raw, syndrome, logical_bits, H_hat, B):
    """Map a raw prediction onto the operator satisfying both hard constraints exactly.

    Solves for the correction that makes `H_hat e0 == [syndrome ; logical_bits]` while
    staying as close to `e_raw` as the affine structure allows:

        e0 = B @ ([s ; logical] + H_hat @ e_raw) + e_raw   (mod 2)

    which satisfies the target because `H_hat B = I`. All inputs are 0/1 tensors; the GF(2)
    products run in float32 regardless of the decoder dtype, since a bfloat16 accumulator
    is only exact to 256 and these sums reach the check degree times the code length.
    """
    work = torch.float32
    e = e_raw.to(work)
    target = torch.cat([syndrome.to(work), logical_bits.to(work)], dim=1)
    residual = (target + e @ H_hat.to(work).t()) % 2
    return ((residual @ B.to(work).t()) + e) % 2


def _cpnd_descent(e0, supports, weights, passes=1):
    """Greedily walk `e0 + ker(H_hat)` towards a lighter operator.

    Input:
        e0: [B, n] projected operator, 0/1
        supports: list of index tensors, one per kernel basis vector, holding its support
        weights: [B, n] per-qubit cost of setting a bit (`w_i = l_v_i`; see the section
            comment above)
        passes: sweeps over the basis; each sweep only accepts strict decreases, so the
            objective is monotone and the loop stops early once a sweep changes nothing

    Flipping along a basis vector changes the cost by `sum_{i in supp} sign_i * w_i`, where
    `sign_i = 1 - 2 e_i`, so the move is accepted exactly when that delta is negative.

    Upstream's `sign[mask][:, v] *= -1` is a no-op: `sign[mask]` is a copy under advanced
    indexing, so the in-place multiply never reaches `sign` and every delta after the first
    accepted flip is evaluated against stale signs. Here both `e` and `sign` are written back
    through explicit `__setitem__` calls on the flipped rows.
    """
    e = e0.to(torch.bool)
    one = torch.ones((), dtype=weights.dtype, device=weights.device)
    sign = torch.where(e, -one, one)

    for _ in range(passes):
        moved = False
        for cols in supports:
            delta = (sign[:, cols] * weights[:, cols]).sum(dim=1)
            rows = (delta < 0).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            moved = True
            idx = rows.unsqueeze(1)
            e[idx, cols] = ~e[idx, cols]
            sign[idx, cols] = -sign[idx, cols]
        if not moved:
            break

    return e


class MultiHeadedAttention(nn.Module):
    """Standard multi-head attention; `mask` is True on entries to suppress."""

    def __init__(self, h, d_model, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        if d_model % h != 0:
            raise ValueError(f"d_model <{d_model}> must be divisible by h <{h}>.")
        self.d_k = d_model // h
        self.h = h
        self.linears = _clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # upstream uses a literal -1e9, which overflows the half dtypes
            neg = (
                -1e9
                if scores.dtype in (torch.float32, torch.float64)
                else torch.finfo(scores.dtype).min
            )
            scores = scores.masked_fill(mask, neg)
        p_attn = self.dropout(F.softmax(scores, dim=-1))
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Pre-norm residual wrapper with a separate norm per stream."""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm_q = nn.LayerNorm(size)
        self.norm_l = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, stream):
        norm = self.norm_q if stream == "syndrome" else self.norm_l
        return x + self.dropout(sublayer(norm(x)))


class SLTDLayer(nn.Module):
    """One Syndrome-Logical Transformer Decoder layer.

    `x` is the stream being updated and `mem` supplies the keys/values: passing `mem is x`
    gives syndrome self-attention, passing the syndrome stream while `x` is the logical
    stream gives the S -> L cross-attention. Only the query side is normalised, matching
    upstream.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SLTDLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mem, mask, stream):
        x = self.sublayer[0](x, lambda q: self.self_attn(q, mem, mem, mask), stream)
        return self.sublayer[1](x, self.feed_forward, stream)


class create(nn.Module):
    """SAQ decoder (paper stages 1 and 2) exposed through the syndrilla decoder contract."""

    # what `train_state` writes and `load_train_state` requires back
    TRAIN_STATE_KEYS = ("state_dict", "optimizer", "scheduler")

    def __init__(self, decoder_cfg, **kwargs) -> None:
        """
        Input:
            decoder_cfg: the information that come from config file (yaml)

        Parameters:
            d_model: token embedding width
            N_dec: number of SLTD layers
            h: number of attention heads
            dropout: attention/feed-forward dropout (training only)
            no_mask: >0 disables the topology attention mask (paper ablation)
            cpnd: run stage 3 (constraint projection + nullspace descent); default True
            cpnd_passes: sweeps over the stabilizer basis in the descent; default 1
            lr / weight_decay / min_lr: optimizer settings, see `configure_optimizer`
            checkpoint: optional path to trained weights (state_dict)

            H_matrix: loaded ldpc matrix, either hx or hz, as 2d tensor
            V_c_row / V_c_col: row/column index of the variable nodes of each check node
        """
        super(create, self).__init__()

        logger.info("Creating saq decoder.")

        # Device and dtype come from the same resolver `load_matrices` used, so the
        # decoder cannot land somewhere its own matrices did not: `parse_device_dtype`
        # falls back to cpu when cuda is configured but unavailable, which is what lets
        # a cuda yaml still run on a cpu-only host.
        #
        # dtype defaults to float32 here (not the shared float64) because this is a
        # neural network: float64 attention is several times slower for no accuracy gain.
        self.device, _ = parse_device_dtype(decoder_cfg)
        self.dtype = decoder_cfg.get("dtype", "float32")
        if self.dtype not in {"float32", "float64", "bfloat16", "float16"}:
            logger.warning(
                f"Invalid input data type <{self.dtype}>, default to <torch.float32>."
            )
            self.dtype = "float32"
        self.dtype = torch.__dict__[self.dtype]

        self.batch_size = 1

        self.check_type = decoder_cfg.get("check_type", "hx")
        if self.check_type.lower() not in {"hx", "hz"}:
            logger.warning(
                f"Invalid input check type <{self.check_type}>, default to <hx>."
            )
            self.check_type = "hx"

        if "code_type" in decoder_cfg:
            raise ValueError(
                "Decoder <saq>: <code_type> was removed. The code family is measured "
                "from the parity-check matrix, so nothing has to declare it; delete "
                "the key."
            )

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "saq requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        self.H_shape, V_c_row, V_c_col, H_matrix = bundle.select(self.check_type)
        # a circuit-level detector error model, where a column is a fault mechanism
        # rather than a qubit, so the code-family relations below do not apply to it
        source = (
            bundle.Hx_matrix if self.check_type.lower() == "hx" else bundle.Hz_matrix
        )
        self.from_circuit_dem = getattr(source, "is_circuit_dem", False)

        l_matrix = (
            bundle.lx_matrix if self.check_type.lower() == "hx" else bundle.lz_matrix
        )
        l_matrix = torch.as_tensor(np.asarray(l_matrix))

        self.m, self.n = int(self.H_shape[0]), int(self.H_shape[1])
        self.k = int(l_matrix.shape[0])
        self.logical_classes = 2**self.k
        if self.k > 16:
            raise ValueError(
                f"saq enumerates 2^k logical classes; k=<{self.k}> is too large."
            )

        self._check_code_shape(H_matrix)

        self.register_buffer("V_c_row", V_c_row.to(self.device))
        self.register_buffer("V_c_col", V_c_col.to(self.device))
        self.register_buffer("H_matrix", H_matrix.to(self.device))
        # [n, k]: right-multiplying an error vector by this gives its logical syndrome
        self.register_buffer(
            "logic_matrix", l_matrix.t().to(self.device).to(self.dtype)
        )

        # The settings below are grouped in the yaml so each block has one reader.
        # A stale flat yaml would otherwise keep parsing while silently falling back to
        # the defaults for every key it moved.
        for legacy, block in (
            ("d_model", "model"),
            ("N_dec", "model"),
            ("h", "model"),
            ("dropout", "model"),
            ("no_mask", "model"),
            ("cpnd_passes", "cpnd"),
            ("lr", "optimizer"),
            ("weight_decay", "optimizer"),
            ("min_lr", "optimizer"),
        ):
            if legacy in decoder_cfg:
                raise ValueError(
                    f"Decoder <saq>: <{legacy}> moved under the decoder yaml's "
                    f"<config.{block}> block. Nest it there rather than at the top level."
                )
        for legacy in ("lambda_loss_lc", "lambda_loss_lp", "lambda_loss_ent"):
            if legacy in decoder_cfg:
                raise ValueError(
                    f"Decoder <saq>: <{legacy}> moved to the loss yaml as "
                    f"<{legacy.replace('lambda_loss_', 'lambda_')}>. Remove it from the "
                    f"decoder yaml and pass the loss yaml with -ls."
                )

        cpnd_cfg = decoder_cfg.get("cpnd", {})
        if not isinstance(cpnd_cfg, dict):
            raise ValueError(
                f"Decoder <saq>: <config.cpnd> is a block with <enable> and <passes>, got "
                f"<{cpnd_cfg!r}>."
            )
        # CPND is inference only: it is not differentiable and cannot affect the loss,
        # which reads the pre-CPND llr/logits. A decoder built for training therefore
        # skips it outright, precompute included, rather than building algebra it will
        # never run. `main.py` passes the mode; what it means is decided here.
        self.training_mode = bool(kwargs.get("training", False))
        self.use_cpnd = bool(cpnd_cfg.get("enable", True)) and not self.training_mode
        self.cpnd_passes = int(cpnd_cfg.get("passes", 1))
        if self.cpnd_passes < 1:
            logger.warning(
                f"Invalid input passes <{self.cpnd_passes}>, default to <1>."
            )
            self.cpnd_passes = 1
        if self.training_mode and bool(cpnd_cfg.get("enable", True)):
            logger.info("saq is being trained; cpnd is inference only and stays off.")
        if self.use_cpnd:
            self._build_cpnd(H_matrix, l_matrix)

        model_cfg = decoder_cfg.get("model", {})
        if not isinstance(model_cfg, dict):
            raise ValueError(
                f"Decoder <saq>: <config.model> is a block with <d_model>, <N_dec>, <h>, "
                f"<dropout> and <no_mask>, got <{model_cfg!r}>."
            )
        d_model = int(model_cfg.get("d_model", 128))
        n_dec = int(model_cfg.get("N_dec", 6))
        heads = int(model_cfg.get("h", 16))
        dropout = float(model_cfg.get("dropout", 0.0))
        self.N_dec = n_dec
        self.d_model = d_model

        # training-only, so validated in configure_optimizer rather than here
        optimizer_cfg = decoder_cfg.get("optimizer", {})
        if not isinstance(optimizer_cfg, dict):
            raise ValueError(
                f"Decoder <saq>: <config.optimizer> is a block with <lr>, <weight_decay> and "
                f"<min_lr>, got <{optimizer_cfg!r}>."
            )
        self.lr = optimizer_cfg.get("lr")
        self.weight_decay = optimizer_cfg.get("weight_decay")
        self.min_lr = optimizer_cfg.get("min_lr")

        # Stage 1: shallow logical-prior MLP + the two learned token dictionaries
        self.lp_head = nn.Sequential(
            nn.Linear(self.m, 4 * self.m),
            nn.GELU(),
            nn.Linear(4 * self.m, self.logical_classes),
        )
        self.src_embed_S = nn.Parameter(torch.empty(self.m, d_model))
        self.src_embed_L = nn.Parameter(torch.empty(self.logical_classes, d_model))
        self.global_tok = nn.Parameter(torch.randn(1, 1, d_model))

        # Stage 2: N independent SLTD layers plus the stream norms
        attn = MultiHeadedAttention(heads, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.layers = _clones(SLTDLayer(d_model, attn, ff, dropout), n_dec)
        self.SN_norm = nn.LayerNorm(d_model)
        self.LN_norm = nn.LayerNorm(d_model)
        if n_dec > 1:
            self.SN_norm2 = nn.LayerNorm(d_model)
            self.LN_norm2 = nn.LayerNorm(d_model)

        # output heads: token states -> per-qubit LLR / logical class logits
        self.oned_embed_SN = nn.Linear(d_model, 1)
        self.oned_embed_LN = nn.Linear(d_model, 1)
        self.out_fc_S = nn.Linear(self.m, self.n)
        self.out_fc_L = nn.Linear(self.logical_classes, self.logical_classes)

        self._build_masks(H_matrix, no_mask=int(model_cfg.get("no_mask", 0)) > 0)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.to(device=self.device, dtype=self.dtype)
        self._load_checkpoint(decoder_cfg.get("checkpoint"))

        self.algo = "saq"
        # both built by configure_optimizer; None until then, so decoding never carries
        # an optimizer it will not use
        self.optimizer = None
        self.scheduler = None
        # SAQ decodes in one feed-forward pass, so every sample always costs exactly one
        # "iteration"; max_iter from the config does not apply.
        self.max_iter = 1
        self.num_max_iter = 1
        self.i = 0
        if decoder_cfg.get("max_iter") not in (None, 1):
            logger.warning(
                f'saq is single-shot; ignoring max_iter <{decoder_cfg.get("max_iter")}>.'
            )

        logger.info("Complete.")

    def _check_code_shape(self, H_matrix):
        """Warn when the loaded matrix does not look like a code saq is validated on.

        The family is read off the matrix rather than configured, since nothing saq does
        depends on being told: a toric lattice's stabilizers are linearly dependent (all
        checks of one type multiply to the identity), so hx/hz carries exactly one
        redundant row, while a surface code's does not. Both facts are measurable, so a
        `code_type` key could only ever disagree with the matrix it describes.
        """
        if self.from_circuit_dem:
            # a DEM's columns are circuit fault mechanisms, so neither test below means
            # what it does on a code: its rank deficiency counts redundant detectors,
            # not redundant stabilizers, and its column count is a fault count that can
            # solve a code relation by coincidence (a distance-3 rotated circuit over 3
            # rounds gives 221, which is the unrotated relation at distance 11)
            logger.info(
                f"saq is running on a circuit-level detector error model: "
                f"<{self.m}> detectors, <{self.n}> fault mechanisms. The code-family "
                f"checks do not apply and no distance is inferred."
            )
            return
        rank = _rref_gf2(H_matrix.detach().cpu().numpy())[2].size
        deficiency = self.m - rank
        if deficiency not in (0, 1):
            logger.warning(
                f"the {self.check_type} matrix has rank deficiency <{deficiency}>; saq "
                f"is validated on toric (1) and surface (0) codes only."
            )
        if not self._families():
            logger.warning(
                f"<{self.n}> qubits fits no code family saq knows "
                f"({', '.join(self._QUBIT_COUNT)}); its distance is unknown."
            )

    def _build_cpnd(self, H_matrix, l_matrix):
        """Precompute the CPND projection and the stabilizer moves for the descent.

        Everything here depends only on the code, so it is built once at construction:
          - `cpnd_rows`: a maximal independent subset of the stabilizer rows. A toric
            lattice's checks are linearly dependent, and `[H; L]` cannot have full row rank
            (which the right inverse needs) until the redundant ones are dropped. Upstream
            hardcodes `H[:-1]` for toric; picking the independent rows works for both
            families and for whichever rows the loader happens to order last.
          - `cpnd_H_hat = [H[rows]; L]` and its right inverse `cpnd_B`.
          - `cpnd_supports`: the support of each `ker(H_hat)` basis vector, i.e. the qubits
            one stabilizer move flips. Stored as index tensors so the descent gathers only
            the columns it touches.
        """
        H_np = H_matrix.detach().cpu().numpy().astype(np.uint8) % 2
        L_np = l_matrix.detach().cpu().numpy().astype(np.uint8) % 2

        rows = _independent_rows(H_np)
        H_hat = np.vstack([H_np[rows], L_np])
        B = _right_inverse(H_hat)
        basis = _kernel_basis(H_hat)

        self.register_buffer(
            "cpnd_rows",
            torch.as_tensor(rows, dtype=torch.long, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "cpnd_H_hat",
            torch.as_tensor(H_hat, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "cpnd_B",
            torch.as_tensor(B, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.cpnd_supports = [
            torch.as_tensor(
                np.flatnonzero(basis[:, j]), dtype=torch.long, device=self.device
            )
            for j in range(basis.shape[1])
        ]

        dropped = self.m - len(rows)
        logger.info(
            f"CPND ready: dropped <{dropped}> dependent check row(s), "
            f"[H; L] is <{H_hat.shape[0]}>x<{H_hat.shape[1]}>, "
            f"<{len(self.cpnd_supports)}> stabilizer moves, <{self.cpnd_passes}> pass(es)."
        )

    def _build_masks(self, H_matrix, no_mask=False):
        """Topology mask M_S plus the (unrestricted) logical cross-attention mask.

        Two stabilizers are neighbours when they share a qubit, i.e. (H H^T)_{ij} != 0. The
        global token sits at index 0 and is connected to everything, which is what keeps
        long-range information flowing despite the local mask. Masks are stored with
        True = "suppress this attention entry".
        """
        if no_mask:
            self.register_buffer("src_mask_SN", None)
            self.register_buffer("src_mask_LN", None)
            return

        H = H_matrix.detach().cpu().to(torch.float32)
        loc = (H @ H.t()) > 0
        loc.fill_diagonal_(True)

        star = torch.zeros(self.m + 1, self.m + 1, dtype=torch.bool)
        star[1:, 1:] = loc
        star[0, :] = True
        star[:, 0] = True

        self.register_buffer("src_mask_SN", ~star.unsqueeze(0).unsqueeze(0))
        # logical tokens attend over every syndrome token (incl. the global one), so this
        # mask suppresses nothing; it is kept explicit to mirror the paper's formulation.
        self.register_buffer(
            "src_mask_LN",
            torch.zeros(1, 1, self.logical_classes, self.m + 1, dtype=torch.bool),
        )

    def _load_checkpoint(self, path):
        if path is None:
            logger.warning(
                "saq has no `checkpoint`; weights are randomly initialized. That is "
                "expected when training; to decode, train the model first and point "
                "`checkpoint` at the resulting state_dict."
            )
            return
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(
                f"saq checkpoint <{path}> has no weights for <{missing}>. It was saved "
                f"from a different architecture; retrain, or point `checkpoint` at a "
                f"checkpoint matching this decoder yaml."
            )
        if unexpected:
            logger.warning(f"saq checkpoint <{path}>: unexpected keys <{unexpected}>.")
        logger.info(f"Loaded saq weights from <{path}>.")

    def forward(self, io_dict):
        """Single-pass SAQ decoding.

        Input:
            syndrome: measured syndrome, [batch, m] in {0, 1}

        Output:
            e_v: estimated error, [batch, n] in {0, 1}
            llr: per-qubit posterior LLR (positive => no error), [batch, n]
            iter: always 1 -- SAQ does not iterate
            converge: 1 where the estimate reproduces the measured syndrome

        `io_dict['llr0']` (the channel LLR) is not consumed: SAQ conditions on the
        syndrome alone, and the physical error rate is learned from the training
        distribution rather than supplied per shot.
        """
        logger.info("Initializing saq decoding.")

        syndrome = io_dict["synd"].to(device=self.device, dtype=self.dtype)
        self.batch_size = syndrome.size(0)

        # the transformer is trained on the +-1 encoding used by upstream's dataset
        syndrome_pm = bin_to_sign(syndrome)

        logger.info("Complete.")
        logger.info("Starting decoding pass.")

        self.i = 1

        # stage 1: dual-stream representation construction
        out_LP = self.logical_prior(syndrome_pm)
        SN, LN = self.build_streams(syndrome_pm, out_LP)

        # stage 2: syndrome-logical transformer decoder
        for idx in range(self.N_dec):
            SN, LN = self.layer_update(SN, LN, idx)

        l_v, out_L = self.head_update(SN, LN)

        e_v = self.hard_decision(l_v)

        # stage 3: constraint-projected nullspace descent (inference-only post-processing)
        if self.use_cpnd:
            e_v = self.project(e_v, syndrome, out_L)
            e_v = self.nullspace_descent(e_v, l_v)

        s_est = self.syndrome_estimation(e_v)

        converges = torch.all(s_est == syndrome, dim=1).long()
        num_iters = torch.ones([self.batch_size], device=self.device, dtype=torch.long)

        logger.info("Complete.")
        logger.info(f"Converged samples: <{int(converges.sum())}>/<{self.batch_size}>.")
        io_dict.update(
            {
                "e_v": e_v,
                "iter": num_iters,
                "llr": l_v,
                "converge": converges,
                "logical_logits": out_L,
                "logical_prior": out_LP,
            }
        )
        return io_dict

    def logical_prior(self, syndrome):
        """Stage 1a: shallow MLP mapping the syndrome to initial logical class logits.

        Output is [batch, 2^k]. It both seeds the logical token stream and is supervised
        directly by the `L_LP` loss term, which gives the logical stream a usable starting
        point before any attention has run.
        """
        return self.lp_head(syndrome)

    def build_streams(self, syndrome, out_LP):
        """Stage 1b: format conversion (syndrome vector -> dual token streams).

        The SAQ analogue of the BP decoders' `v2c`: it moves the problem out of the raw
        per-check vector layout into the layout the updates operate on. Each stabilizer
        scales its learned embedding by its own +-1 measurement, each logical class scales
        its embedding by the corresponding prior logit, and a learned global token is
        prepended to the syndrome stream as the one node every other node may attend.

        Returns SN [batch, m+1, d] and LN [batch, 2^k, d].
        """
        SN = self.src_embed_S.unsqueeze(0) * syndrome.unsqueeze(-1)
        LN = self.src_embed_L.unsqueeze(0) * out_LP.unsqueeze(-1)
        g = self.global_tok.expand(SN.size(0), -1, -1)
        return torch.cat([g, SN], dim=1), LN

    def sn_update(self, SN, idx):
        """Syndrome-stream update: topology-masked self-attention over the stabilizers.

        Each stabilizer token may only attend its 1-hop neighbours in the Tanner-graph
        projection (checks sharing a qubit) plus the global token, which is the constraint
        that keeps the cost linear in the syndrome size.
        """
        return self.layers[idx](SN, SN, self.src_mask_SN, "syndrome")

    def ln_update(self, LN, SN, idx):
        """Reverse (syndrome -> logical) stream update: unrestricted cross-attention.

        The counterpart of `sn_update`: the logical tokens are the queries and the freshly
        updated syndrome tokens supply keys and values. This direction is deliberately
        left unmasked -- degenerate errors differing by a stabilizer are only separable by
        looking at the whole syndrome at once.
        """
        return self.layers[idx](LN, SN, self.src_mask_LN, "logical")

    def layer_update(self, SN, LN, idx):
        """One full SLTD layer: token update of both streams.

        Runs the syndrome self-attention first and feeds its result straight into the
        reverse cross-attention, so the logical stream always reads the current syndrome
        representation rather than the previous layer's. At the mid-depth layer both
        streams are re-normalised, which upstream uses to keep the residual scale in check
        for deep stacks.
        """
        SN = self.sn_update(SN, idx)
        LN = self.ln_update(LN, SN, idx)
        if self.N_dec > 1 and idx == self.N_dec // 2:
            SN = self.SN_norm2(SN)
            LN = self.LN_norm2(LN)
        return SN, LN

    def head_update(self, SN, LN):
        """Output heads: token states -> per-qubit LLR and logical class logits.

        The reverse of `build_streams` (the SAQ analogue of the BP decoders' `c2v`): each
        stream is normalised, collapsed to one scalar per token, then linearly mapped --
        the m syndrome scalars to the n qubit LLRs, and the logical scalars to refined
        class logits. The qubit output follows the same sign convention as the BP
        decoders' posterior LLR: positive means the qubit is more likely error-free.
        """
        SN = self.SN_norm(SN)
        LN = self.LN_norm(LN)
        l_v = self.out_fc_S(self.oned_embed_SN(SN[:, 1:, :]).squeeze(-1))
        out_L = self.out_fc_L(self.oned_embed_LN(LN).squeeze(-1))
        return l_v, out_L

    def project(self, e_raw, syndrome, out_L):
        """Stage 3a: project the hard decision onto the exactly-feasible operator.

        The transformer output need not satisfy `H e = s` at all. This maps it to the
        operator that reproduces the measured syndrome *and* the logical class the
        transformer predicted, using the precomputed right inverse of `[H; L]`.

        Only the independent check rows constrain the projection (see `_build_cpnd`). Under
        perfect measurement the dropped rows are implied by the kept ones, so the result also
        satisfies the full `H`; under noisy syndrome extraction a syndrome outside the image
        of `H` cannot be met and those samples fall out as unconverged, since `forward`
        re-checks against the full matrix.
        """
        logical_bits = _logits_to_logical_bits(out_L, self.k)
        e0 = _cpnd_project(
            e_raw,
            syndrome[:, self.cpnd_rows],
            logical_bits,
            self.cpnd_H_hat,
            self.cpnd_B,
        )
        return e0.to(self.dtype)

    def nullspace_descent(self, e0, l_v):
        """Stage 3b: lighten the operator without leaving its constraint coset.

        Walks the `ker([H; L])` basis greedily, accepting any move that lowers
        `sum_i e_i * l_v_i`. Because the kernel is of `[H; L]` rather than of `H`, each move
        is a stabilizer: the syndrome and the logical class both survive, and only the
        representative changes.

        The weights are the true posterior LLRs, which is the negative log-likelihood cost of
        setting each qubit under this repo's sign convention. Upstream weights by
        `-log(p/(1-p))` of an already-binarised output, which both inverts the sign and
        discards every magnitude; the CPND section comment above documents why.
        """
        e = _cpnd_descent(e0, self.cpnd_supports, l_v.detach(), passes=self.cpnd_passes)
        return e.to(self.dtype)

    def hard_decision(self, l_v):
        """Hard decision: map posterior LLRs to a binary error estimate.

        A non-positive LLR (<= 0) means the qubit is more likely flipped, so it is set to
        1; otherwise 0. Returns the estimate in the decoder's dtype.
        """
        return torch.where(l_v <= 0.0, 1.0, 0.0).to(self.dtype)

    def syndrome_estimation(self, e_v):
        """Syndrome the estimated error would have produced, as H @ e_v over GF(2).

        `V_c_col` is ragged-padded with the index `n`, so a zero dummy column is appended
        before gathering and the padding contributes nothing to the parity.
        """
        dummy = torch.zeros([e_v.size(0), 1], dtype=e_v.dtype, device=e_v.device)
        temp_e = torch.cat([e_v, dummy], dim=1)
        estimated_syndrome = temp_e[:, self.V_c_col].sum(dim=2).to(dtype=self.dtype)
        return torch.where((estimated_syndrome % 2) > 0.0, 1.0, 0.0)

    def configure_optimizer(self, epochs):
        """Adam plus a cosine schedule over `epochs`, from this decoder's own config.

        Both are stored on the decoder rather than returned: `update` steps the optimizer
        and `lr_step` the schedule, so the training loop never handles either directly.
        The gradients are zeroed here so that on entry to every `backward` they are clean,
        which is the invariant the old zero_grad-before-backward ordering gave for free.
        """
        lr = _require_number(self.lr, "lr")
        weight_decay = _require_number(self.weight_decay, "weight_decay")
        min_lr = _require_number(self.min_lr, "min_lr")

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)
        self.optimizer.zero_grad(set_to_none=True)

    def train_fingerprint(self):
        """The model half of a resume fingerprint: what this decoder is.

        The decoder states its own algorithm, the code shape it was built for, and the
        optimizer settings `configure_optimizer` will read. The schedule half (epochs,
        batches, seed, batch size) belongs to `MetricState`, which merges the two, so
        neither side has to reach into the other.
        """
        return {
            "algo": self.algo,
            "n": self.n,
            "m": self.m,
            "k": self.k,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "min_lr": self.min_lr,
        }

    def check_train_batch(self, rounds, number_channel):
        """Training stage: reject a batch shape this decoder cannot learn from.

        This is saq's constraint, not training's, which is why it lives here rather than
        in the loop. A decoder that consumed the rounds dimension itself would have no
        reason to refuse one.

        `rounds > 1`: `RoundFlattenWrapper` folds rounds into the batch and unfolds
        `e_v` / `synd` / `llr` / `converge` / `iter` afterwards, but not
        `logical_logits` / `logical_prior`, which only this decoder writes. The loss
        would then be handed an llr at [B, d, n] and a logical head at [B*d, 2^k].

        `number_channel > 1`: saq is not in the wrapper's multi-channel set, so a second
        channel is read as a second round and flattened the same way. It is built from a
        single check type's H regardless.

        The loss shares both assumptions, since it is what pairs the llr with the
        logical head; a loss written against a different output contract would need its
        own check.
        """
        if rounds != 1:
            raise ValueError(
                f"Decoder <{self.algo}> trains on one syndrome measurement round at a "
                f"time, got rounds <{rounds}>. Its logical head is written per forward "
                f"row and is not unfolded back over the rounds dimension, so the loss "
                f"would pair mismatched shapes. Train with a single-round measurer."
            )
        if number_channel != 1:
            raise ValueError(
                f"Decoder <{self.algo}> trains on a single check type, got "
                f"number_channel <{number_channel}>. It is built from one H matrix, and "
                f"a second channel is read as a second round. Train with "
                f"number_channel 1."
            )

    def backward(self, loss):
        """Training stage: accumulate this batch's gradients, if this batch trains.

        The autograd counterpart of `forward`. Paired one-to-one with `update`; the loop
        does not accumulate over several batches.

        A validation batch is skipped here rather than in the loop. `set_training` has
        already put this decoder in the mode its batch runs in, so the mode is the
        answer to whether a gradient step belongs to it -- asking the schedule a second
        time at the call site is a second answer that can disagree with this one. It
        cannot be left to fall through either: a validation batch is built with grad
        off, so `loss.backward()` raises rather than quietly doing nothing.
        """
        if not self.training:
            return
        loss.backward()

    def update(self):
        """Training stage: apply one optimizer step, then reset the gradients.

        Skipped on a validation batch for the reason given in `backward`, which this is
        paired with: there are no gradients to apply, and applying the previous training
        batch's would be worse than doing nothing.

        Resetting here rather than before the next `backward` keeps the loop reading
        backward -> update, and `configure_optimizer` establishes the same clean-gradient
        precondition for the very first step.
        """
        if not self.training:
            return
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def lr_step(self):
        """Advance the cosine schedule. Once per epoch, not once per batch."""
        self.scheduler.step()

    def current_lr(self):
        """The learning rate the next `update` will use."""
        return self.scheduler.get_last_lr()[0]

    def set_training(self, training):
        """Switch between accumulating gradients and not, for one batch.

        The decoder owns whether it is building a graph, so it sets both the module mode
        and the global grad switch rather than leaving the caller to keep them in step.
        """
        self.train(training)
        torch.set_grad_enabled(training)

    def train_state(self):
        """Everything this decoder needs to resume a run, not just to decode one.

        Weights alone describe a *trained* decoder; they do not describe a *training*
        one. Adam's per-parameter moments and the cosine schedule's position decide the
        size and direction of the next step, so reloading without them warm-starts a new
        run instead of continuing the old one.
        """
        if self.optimizer is None:
            raise ValueError(
                "saq has no optimizer to save; call configure_optimizer() first."
            )
        return {
            "state_dict": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": self._rng_state(),
        }

    def _rng_state(self):
        """The generators a resumed run must carry on drawing from.

        Which generators there are is a question about *this decoder's device*, so it
        is answered here rather than by the training loop probing `torch.cuda`. The
        error stream is global state, but the device it is drawn on is the decoder's.
        """
        state = {"cpu": torch.get_rng_state()}
        if str(self.device).startswith("cuda"):
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _load_rng_state(self, state):
        """Put the draw sequence back where the interrupted run left it.

        The error stream itself no longer depends on this: `MetricState.begin_batch`
        reseeds at every phase boundary, including after a resume, so epoch N draws the
        same errors whichever way it was reached. This restores the rest of the
        generator state for anything that resumes mid-epoch.
        """
        if not state:
            return
        # `main.py` loads the checkpoint with map_location=<this decoder's device>, so on
        # a GPU run every saved tensor arrives on cuda -- including these. Both setters
        # want the state back on the CPU as a ByteTensor, hence the `.cpu()` on each.
        torch.set_rng_state(state["cpu"].cpu())
        if "cuda" in state and str(self.device).startswith("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])

    def load_train_state(self, state):
        """Restore what `train_state` saved, onto an already configured optimizer.

        `configure_optimizer` must have run: Adam's moments are keyed by parameter, so
        there has to be an optimizer holding this module's parameters to load them into.
        """
        if self.optimizer is None:
            raise ValueError(
                "saq cannot resume before configure_optimizer(); there is no optimizer "
                "to restore the saved moments into."
            )
        missing = [key for key in self.TRAIN_STATE_KEYS if key not in state]
        if missing:
            raise ValueError(
                f"saq training checkpoint is missing <{', '.join(missing)}>. It was "
                f"written by an older version that saved weights only; that file can "
                f"still be decoded from via the decoder yaml's `checkpoint` key, but a "
                f"run cannot be resumed from it."
            )
        self.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self._load_rng_state(state.get("rng"))

    def checkpoint_stem(self):
        """The name this run's weights are saved under: algo, check type, code size.

        A run directory holding one `best.pt` cannot hold two configurations, so the
        name carries what distinguishes them instead.
        """
        return f"{self.algo}_{self.check_type}_{self._code_size()}"

    # How each code family fixes its qubit count from its distance. Distance is not a
    # configured value anywhere, so it is solved back out of `n` when one of these fits.
    _QUBIT_COUNT = {
        "toric": lambda d: 2 * d * d,
        "rotated_surface": lambda d: d * d,
        "unrotated_surface": lambda d: d * d + (d - 1) * (d - 1),
    }

    def _code_size(self):
        """`d<distance>` when a known code relation fixes it from `n`, else `n<n>`.

        The declared family is tried first and the others follow, because a matrix can
        belong to a family the yaml does not name: the shipped `surface_*` matrices are
        unrotated (41 qubits at distance 5) while the yaml calls them rotated. A count
        matching no relation leaves the distance unknown rather than guessed, since a
        wrong distance in a filename outlives the run that wrote it.
        """
        if self.from_circuit_dem:
            # named for what actually distinguishes two DEM runs. A code distance would
            # be a coincidence of the fault count here, and a wrong distance in a
            # filename outlives the run that wrote it
            return f"dem{self.m}x{self.n}"
        fits = self._families()
        return f"d{fits[0][1]}" if fits else f"n{self.n}"

    def _families(self):
        """`(family, distance)` for every known code whose relation `self.n` solves.

        Almost always one entry. A count can satisfy two relations, `n = 25` being both
        a rotated surface code at distance 5 and an unrotated one at distance 4, in
        which case the first is taken: the families are tried in the fixed order they
        are declared, so the same matrix always yields the same name.
        """
        return [
            (family, d)
            for family in self._QUBIT_COUNT
            if (d := self._solve_distance(family)) is not None
        ]

    def _solve_distance(self, family):
        """The distance a family would need to have `self.n` qubits, or None."""
        qubit_count = self._QUBIT_COUNT.get(family)
        if qubit_count is None:
            return None
        d = 1
        while qubit_count(d) < self.n:
            d += 1
        return d if qubit_count(d) == self.n else None

    def save_checkpoint(self, path):
        """Write this decoder's weights, in the form `_load_checkpoint` reads back."""
        torch.save(self.state_dict(), path)

    def save_checkpoints(self, run_dir, is_best, extra=None):
        """Write `<stem>_last.pt`, and `<stem>.pt` too when validation improved.

        What a checkpoint is belongs to the decoder; whether this epoch improved belongs
        to the metrics, which pass that answer in.

        The two files differ on purpose. `<stem>.pt` stays a bare state_dict: it is what
        a decoder yaml's `checkpoint` key points at, and decoding has no use for
        optimizer moments. `<stem>_last.pt` carries the full training state plus
        `extra`, the part of the run position the caller owns (epoch, best-so-far,
        history, RNG, fingerprint). `_load_checkpoint` unwraps the `state_dict` key, so
        the resume file decodes too. Both are named after the configuration that
        produced them, so training a second one into the same run directory adds files
        rather than overwriting the first.
        """
        stem = self.checkpoint_stem()
        if is_best:
            self.save_checkpoint(os.path.join(run_dir, f"{stem}.pt"))
        torch.save(
            {**self.train_state(), **(extra or {})},
            os.path.join(run_dir, f"{stem}_last.pt"),
        )
