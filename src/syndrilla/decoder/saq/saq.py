"""SAQ -- Stabilizer-Aware Quantum error correction decoder.

PyTorch port of https://github.com/DavidZenati/SAQ-Decoder (Zenati & Nachmani,
"SAQ: Stabilizer-Aware Quantum Error Correction Decoder", ICLR 2026, arXiv:2512.08914),
adapted to the syndrilla decoder contract.

SAQ is a *learned* decoder: one feed-forward pass maps a syndrome to an error estimate,
so there is no message passing and no per-sample iteration count. `forward` chains the
blocks of the paper's figure 1: `initial_embedding_layer` for stage 1 (dual token streams
from the syndrome and a shallow logical prior), `SAQ_decoder_layer` for stage 2 (N transformer
layers, topology-masked syndrome self-attention plus unrestricted S -> L cross-attention),
then `output_layer` for stage 3 (the heads). Stage 4, CPND, is `project` and
`nullspace_descent`, and runs at inference only.

Samples whose estimate does not reproduce the measured syndrome are reported unconverged,
so a chained decoder such as `osd_0` can retry them. The objective lives in
`syndrilla/loss/logical_centric/` and reads the `llr`, `logical_logits` and
`logical_prior` entries `forward` writes; trained weights come back through the
`checkpoint` config key. Supported codes: toric and surface, rotated or not, plus
circuit-level detector error models; nothing has to declare which, the decoder reads
only the shape of the matrix it is handed.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR

from syndrilla.utils import parse_device_dtype


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


def _rref_gf2(matrix):
    """Reduced row echelon form over GF(2).

    Returns `(R, T, pivots)` with `T @ matrix == R` (mod 2), `T` invertible, and
    `pivots.size` the rank. Eliminates a whole column of rows per pivot, unlike
    `syndrilla.utils.row_echelon`, whose Python loop is too slow at construction.
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


def _right_inverse(H_hat):
    """`B` of shape [n, r] with `H_hat @ B == I_r` over GF(2).

    `R[:, P] = I` at the pivot columns makes `T = H_hat[:, P]^-1`, so scattering `T` into
    the pivot rows of a zero matrix gives a right inverse. Raises rather than returning a
    matrix that silently fails the identity.
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
    """Basis of `ker(H_hat)`, one basis vector per column of the returned [n, g] matrix.

    The reduced-echelon generators, matching upstream: correct, but not minimal weight, so
    the descent takes longer moves than the code's own plaquette/star stabilizers would.
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
    """Class logits [B, 2^k] -> the class's bit pattern [B, k], LSB first.

    Inverse of the packing the logical-centric loss trains against.
    """
    index = logits.argmax(dim=1)
    shifts = torch.arange(k, device=index.device)
    return (index.unsqueeze(1) >> shifts) & 1


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

    `x` is the stream being updated and `mem` supplies keys/values, so `mem is x` gives
    self-attention and passing the syndrome stream gives S -> L cross-attention. Only the
    query side is normalised, matching upstream.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SLTDLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mem, mask, stream):
        x = self.sublayer[0](x, lambda q: self.self_attn(q, mem, mem, mask), stream)
        return self.sublayer[1](x, self.feed_forward, stream)


class create(nn.Module):
    """SAQ decoder exposed through the syndrilla decoder contract."""

    # what `train_state` writes and `load_train_state` requires back
    TRAIN_STATE_KEYS = ("state_dict", "optimizer", "scheduler")

    def __init__(self, decoder_cfg, **kwargs) -> None:
        """
        Input:
            decoder_cfg: the information that come from config file (yaml)

        Settings, by the yaml block that carries them:
            model:     d_model (token width), N_dec (SLTD layers), h (attention heads),
                       dropout, no_mask (>0 disables the topology mask, paper ablation)
            cpnd:      enable (default True, forced off when training), passes (default 1)
            optimizer: lr / weight_decay / min_lr, read by `configure_optimizer`
            top level: check_type, dtype, device, checkpoint

        Matrices arrive through the `bundle` kwarg, not the config.
        """
        super(create, self).__init__()

        logger.info("Creating saq decoder.")

        self.device, _ = parse_device_dtype(decoder_cfg)
        self.dtype = decoder_cfg.get("dtype", "float32")
        if self.dtype not in {"float32", "float64", "bfloat16", "float16"}:
            logger.warning(
                f"Invalid input data type <{self.dtype}>, default to <torch.float32>."
            )
            self.dtype = "float32"
        self.dtype = torch.__dict__[self.dtype]

        self.check_type = decoder_cfg.get("check_type", "hx")
        if self.check_type.lower() not in {"hx", "hz"}:
            logger.warning(
                f"Invalid input check type <{self.check_type}>, default to <hx>."
            )
            self.check_type = "hx"

        if "code_type" in decoder_cfg:
            raise ValueError(
                "Decoder <saq>: <code_type> was removed. Nothing in the decoder branches "
                "on the code family, so nothing has to declare it; delete the key."
            )

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "saq requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        H_shape, _, V_c_col, H_matrix = bundle.select(self.check_type)
        # a circuit-level DEM's columns are fault mechanisms rather than qubits, which is
        # what `metric` names the result file by
        source = (
            bundle.Hx_matrix if self.check_type.lower() == "hx" else bundle.Hz_matrix
        )
        self.from_circuit_dem = getattr(source, "is_circuit_dem", False)

        l_matrix = (
            bundle.lx_matrix if self.check_type.lower() == "hx" else bundle.lz_matrix
        )
        l_matrix = torch.as_tensor(np.asarray(l_matrix))

        self.m, self.n = int(H_shape[0]), int(H_shape[1])
        self.k = int(l_matrix.shape[0])
        self.logical_classes = 2**self.k
        if self.k > 16:
            raise ValueError(
                f"saq enumerates 2^k logical classes; k=<{self.k}> is too large."
            )

        self.register_buffer("V_c_col", V_c_col.to(self.device))
        self.register_buffer("H_matrix", H_matrix.to(self.device))
        self.register_buffer(
            "logic_matrix", l_matrix.t().to(self.device).to(self.dtype)
        )

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

        training = bool(kwargs.get("training", False))
        cpnd_wanted = bool(cpnd_cfg.get("enable", True))
        self.use_cpnd = cpnd_wanted and not training
        self.cpnd_passes = int(cpnd_cfg.get("passes", 1))
        if self.cpnd_passes < 1:
            logger.warning(
                f"Invalid input passes <{self.cpnd_passes}>, default to <1>."
            )
            self.cpnd_passes = 1
        if training and cpnd_wanted:
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

        # stage 1: Initial Embedding Layer
        self.MLP = nn.Sequential(
            nn.Linear(self.m, 4 * self.m),
            nn.GELU(),
            nn.Linear(4 * self.m, self.logical_classes),
        )
        self.learnable_embed_S = nn.Parameter(torch.empty(self.m, d_model))
        self.learnable_embed_L = nn.Parameter(
            torch.empty(self.logical_classes, d_model)
        )
        self.global_tok = nn.Parameter(torch.randn(1, 1, d_model))

        # stage 2: N independent SLTD layers plus the mid-depth re-normalization
        mhca = MultiHeadedAttention(heads, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.layers = _clones(SLTDLayer(d_model, mhca, ff, dropout), n_dec)
        if n_dec > 1:
            self.SN_norm2 = nn.LayerNorm(d_model)
            self.LN_norm2 = nn.LayerNorm(d_model)

        # stage 3 output layer, the final norm of each stream included
        self.SN_norm = nn.LayerNorm(d_model)
        self.LN_norm = nn.LayerNorm(d_model)
        self.proj_e = nn.Linear(d_model, 1)
        self.proj_l = nn.Linear(d_model, 1)
        self.out_fc_S = nn.Linear(self.m, self.n)
        self.out_fc_L = nn.Linear(self.logical_classes, self.logical_classes)

        self._build_masks(H_matrix, no_mask=int(model_cfg.get("no_mask", 0)) > 0)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.to(device=self.device, dtype=self.dtype)
        self._load_checkpoint(decoder_cfg.get("checkpoint"))

        self.algo = "saq"
        # both built by configure_optimizer, so decoding never carries one it will not use
        self.optimizer = None
        self.scheduler = None
        # one feed-forward pass per sample; `num_max_iter` is what `main.py` reads
        self.num_max_iter = 1
        if decoder_cfg.get("max_iter") not in (None, 1):
            logger.warning(
                f'saq is single-shot; ignoring max_iter <{decoder_cfg.get("max_iter")}>.'
            )

        logger.info("Complete.")

    def _build_cpnd(self, H_matrix, l_matrix):
        """Precompute `[H; L]`, its right inverse, and the stabilizer moves.

        All of it depends only on the code, so it is built once. Dropping the dependent
        check rows is what lets `[H; L]` reach the full row rank the right inverse needs;
        upstream hardcodes `H[:-1]` for toric, which only works for toric.
        """
        H_np = H_matrix.detach().cpu().numpy().astype(np.uint8) % 2
        L_np = l_matrix.detach().cpu().numpy().astype(np.uint8) % 2

        # column pivots of H^T are row pivots of H
        rows = _rref_gf2(H_np.T)[2]
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

        # nullspace_descent support 
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

        Two stabilizers are neighbours when they share a qubit, `(H H^T)_{ij} != 0`. The
        global token at index 0 is connected to everything, which is what keeps long-range
        information flowing. True = "suppress this attention entry".
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
        # suppresses nothing; kept explicit to mirror the paper's formulation
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
            synd: measured syndrome, [batch, m] in {0, 1}

        Output:
            e_v: estimated error, [batch, n] in {0, 1}
            llr: per-qubit posterior LLR (positive => no error), [batch, n]
            iter: always 1 -- SAQ does not iterate
            converge: 1 where the estimate reproduces the measured syndrome
        """
        logger.info("Starting SAQ decoding pass.")

        syndrome = io_dict["synd"].to(device=self.device, dtype=self.dtype)
        batch = syndrome.size(0)

        # Initial Embedding Layer
        out_LP, SN, LN = self.initial_embedding_layer(syndrome)

        # SAQ decoder layer
        SN, LN = self.SAQ_decoder_layer(SN, LN)

        # output layer
        l_v, out_L = self.output_layer(SN, LN)
        e_v = torch.where(l_v <= 0.0, 1.0, 0.0).to(self.dtype)

        # only cpnd used
        if self.use_cpnd:
            e_v = self.project(e_v, syndrome, out_L)
            e_v = self.nullspace_descent(e_v, l_v)

        s_est = self.syndrome_estimation(e_v)

        converges = torch.all(s_est == syndrome, dim=1).long()
        num_iters = torch.ones([batch], device=self.device, dtype=torch.long)

        logger.info("Complete.")
        logger.info(f"Converged samples: <{int(converges.sum())}>/<{batch}>.")
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

    def initial_embedding_layer(self, syndrome):
        """Stage 1: Initial Embedding Layer. Takes the measured syndrome in {0, 1}."""
        syndrome_pm = 1 - 2 * syndrome
        out_LP = self.MLP(syndrome_pm)
        SN = self.learnable_embed_S.unsqueeze(0) * syndrome_pm.unsqueeze(-1)
        LN = self.learnable_embed_L.unsqueeze(0) * out_LP.unsqueeze(-1)
        SN = torch.cat([self.global_tok.expand(SN.size(0), -1, -1), SN], dim=1)
        return out_LP, SN, LN

    def SAQ_decoder_layer(self, SN, LN):
        """Stage 2: SAQ Decoder Layer."""
        for idx in range(self.N_dec):
            SN = self.layers[idx](SN, SN, self.src_mask_SN, "syndrome")
            LN = self.layers[idx](LN, SN, self.src_mask_LN, "logical")
            if self.N_dec > 1 and idx == self.N_dec // 2:
                SN = self.SN_norm2(SN)
                LN = self.LN_norm2(LN)
        return SN, LN

    def output_layer(self, SN, LN):
        """Stage 3: Output Layer."""
        l_v = self.out_fc_S(self.proj_e(self.SN_norm(SN)[:, 1:, :]).squeeze(-1))
        out_L = self.out_fc_L(self.proj_l(self.LN_norm(LN)).squeeze(-1))
        return l_v, out_L

    def project(self, e_raw, syndrome, out_L):
        """Stage 4a: map the hard decision onto the exactly-feasible operator.

            e0 = B @ ([s ; logical] + H_hat @ e_raw) + e_raw   (mod 2)

        which satisfies `H e = s` and `L e = class` because `H_hat B = I`. Only the
        independent check rows constrain it; under perfect measurement the dropped rows
        follow, and under noisy extraction a syndrome outside the image of `H` falls out
        as unconverged when `forward` re-checks against the full matrix.

        The GF(2) products run in float32 whatever the decoder dtype: a bfloat16
        accumulator is exact only to 256, and these sums reach check degree times `n`.
        """
        logical_bits = _logits_to_logical_bits(out_L, self.k)

        work = torch.float32
        e = e_raw.to(work)
        target = torch.cat(
            [syndrome[:, self.cpnd_rows].to(work), logical_bits.to(work)], dim=1
        )
        residual = (target + e @ self.cpnd_H_hat.to(work).t()) % 2
        e0 = ((residual @ self.cpnd_B.to(work).t()) + e) % 2
        return e0.to(self.dtype)

    def nullspace_descent(self, e0, l_v):
        """Stage 4b: lighten the operator without leaving its constraint coset.

        A move along a basis vector changes the cost by `sum_{i in supp} sign_i * l_v_i`
        with `sign_i = 1 - 2 e_i`, and is taken when that is negative. Sweeps accept only
        strict decreases, so the objective is monotone and a sweep that changes nothing
        ends the loop.

        Upstream's `sign[mask][:, v] *= -1` is a no-op, since `sign[mask]` is a copy under
        advanced indexing, leaving every later delta on stale signs; both `e` and `sign`
        are written back explicitly here.
        """
        weights = l_v.detach()
        e = e0.to(torch.bool)
        one = torch.ones((), dtype=weights.dtype, device=weights.device)
        sign = torch.where(e, -one, one)

        for _ in range(self.cpnd_passes):
            moved = False
            for cols in self.cpnd_supports:
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

        return e.to(self.dtype)

    def syndrome_estimation(self, e_v):
        """`H @ e_v` over GF(2).

        `V_c_col` is ragged-padded with the index `n`, so a zero dummy column is appended
        and the padding contributes nothing to the parity.
        """
        dummy = torch.zeros([e_v.size(0), 1], dtype=e_v.dtype, device=e_v.device)
        temp_e = torch.cat([e_v, dummy], dim=1)
        estimated_syndrome = temp_e[:, self.V_c_col].sum(dim=2).to(dtype=self.dtype)
        return torch.where((estimated_syndrome % 2) > 0.0, 1.0, 0.0)

    def configure_optimizer(self, epochs):
        """Adam plus a cosine schedule over `epochs`, from this decoder's own config.

        Both are stored on the decoder: the training loop steps `self.optimizer` per
        batch and `self.scheduler` per epoch. Zeroing here gives every backward pass
        clean gradients on entry, including the first.
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
        """The model half of a resume fingerprint. `MetricState` owns the schedule half."""
        return {
            "algo": self.algo,
            "n": self.n,
            "m": self.m,
            "k": self.k,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "min_lr": self.min_lr,
        }

    def train_state(self):
        """Everything needed to resume a run, not just to decode one.

        Adam's moments and the schedule position decide the next step's size and
        direction, so weights alone warm-start rather than continue. Which `rng`
        generators exist is a question about this decoder's device, so it is answered here.
        """
        if self.optimizer is None:
            raise ValueError(
                "saq has no optimizer to save; call configure_optimizer() first."
            )
        rng = {"cpu": torch.get_rng_state()}
        if str(self.device).startswith("cuda"):
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {
            "state_dict": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": rng,
        }

    def load_train_state(self, state):
        """Restore what `train_state` saved, onto an already configured optimizer.

        The error stream does not depend on the restored `rng`, since
        `MetricState.train_set_hyperparameter` reseeds at every phase boundary; what is restored
        is the rest of the generator state, for anything resuming mid-epoch.
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
        rng = state.get("rng")
        if rng:
            # map_location puts every saved tensor on this decoder's device; both setters
            # want the state back on the CPU as a ByteTensor
            torch.set_rng_state(rng["cpu"].cpu())
            if "cuda" in rng and str(self.device).startswith("cuda"):
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
