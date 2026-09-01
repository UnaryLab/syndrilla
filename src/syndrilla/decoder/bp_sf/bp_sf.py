import itertools
import math
import random

import torch
from loguru import logger

from syndrilla.decoder.decoder import RebatchSpeedup


def sample_n_choose_k(iterable, k, num_samples):
    """Sample ``num_samples`` weight-``k`` combinations of ``iterable``.

    Faithful transformation of ``mybp.sample_n_choose_k`` from the BP-SF reference: if the
    requested count meets/exceeds the number of combinations, return them all;
    if the sampling fraction is tiny, draw with replacement (collisions are
    vanishingly unlikely); otherwise draw a uniform subset without replacement.
    """
    if k > len(iterable):
        raise ValueError("k cannot be greater than the length of the iterable")

    iterable = list(iterable)
    num_comb = math.comb(len(iterable), k)
    if num_samples >= num_comb:
        return list(itertools.combinations(iterable, k))

    if num_comb > 0 and num_samples / num_comb < 0.001:
        return [tuple(random.sample(iterable, k)) for _ in range(num_samples)]

    return random.sample(list(itertools.combinations(iterable, k)), num_samples)


class create(torch.nn.Module):
    """BP-SF decoder (normalized min-sum BP + syndrome-flipping post-processing).

    The belief-propagation core is identical to ``bp_norm_min_sum`` (parallel
    normalized min-sum with the adaptive factor ``beta = 1 - 2^-i``, which is what
    the reference's ``bp_method="ms", ms_scaling_factor=0`` resolves to). On top of
    it BP-SF adds the two ingredients from https://github.com/Dies-Irae/BP-SF
    ("Fully Parallelized BP Decoding for Quantum LDPC Codes Can Outperform BP-OSD",
    HPCA 2026):

      * **Oscillation tracking** -- during the BP iterations, count per bit how many
        times its hard decision flips relative to the previous iteration. Bits that
        oscillate the most are the ones BP is least sure about.

      * **Syndrome flipping (SF)** -- for any sample BP fails to converge, take the
        ``topk`` most-oscillating bits, try flipping weight-``w`` (``w_min..w_max``)
        combinations of them, adjust the syndrome accordingly (``s' = s XOR H[:,flip]``),
        re-run BP, and on convergence flip those bits back in the estimate.
    """

    def __init__(self, decoder_cfg, **kwargs) -> None:
        super(create, self).__init__()

        logger.info("Creating bp_sf decoder.")

        # set up default device
        device_cfg = decoder_cfg.get("device", {})
        self.device = device_cfg.get(
            "device_type", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device not in {
            "cuda",
            "cpu",
            torch.device("cuda"),
            torch.device("cpu"),
        }:
            logger.warning(
                f"Invalid input device <{self.device}>, default to avaliable device in your machine."
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            device_idx = device_cfg.get("device_idx", 0)
            if device_idx >= torch.cuda.device_count():
                logger.warning(
                    f"Invalid input device index <{device_idx}>, default to avaliable device in your machine."
                )
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(f"cuda:{device_idx}")

        # set up default max_iter
        self.max_iter = decoder_cfg.get("max_iter", 50)
        if self.max_iter <= 0 or not isinstance(self.max_iter, int):
            logger.warning(
                f"Invalid input maximum iteration <{self.max_iter}>, default to <50>."
            )
            self.max_iter = 50

        # set up default dtype
        self.dtype = decoder_cfg.get("dtype", "float64")
        if self.dtype not in {"float32", "float64", "bfloat16", "float16"}:
            logger.warning(
                f"Invalid input data type <{self.dtype}>, default to <torch.float64>."
            )
            self.dtype = "float64"
        self.dtype = torch.__dict__[self.dtype]

        self.batch_size = 1

        self.check_type = decoder_cfg.get("check_type", "hx")
        if self.check_type.lower() not in {"hx", "hz"}:
            logger.warning(
                f"Invalid input check type <{self.check_type}>, default to <hx>."
            )
            self.check_type = "hx"

        bundle = kwargs.get("bundle")
        if bundle is None:
            raise ValueError(
                "bp_sf requires a pre-loaded MatrixBundle via the `bundle` kwarg."
            )
        self.Hx_matrix = bundle.Hx_matrix
        self.Hz_matrix = bundle.Hz_matrix
        self.lx_matrix = bundle.lx_matrix
        self.lz_matrix = bundle.lz_matrix
        self.H_shape, self.V_c_row, self.V_c_col, self.H_matrix = bundle.select(
            self.check_type
        )

        self.mask_dummy = self.V_c_col == self.H_shape[1]

        # dense parity-check matrix [M, N], used to compute the syndrome shift of a
        # candidate flip set during SF post-processing.
        self.H_dense = self.H_matrix.to(dtype=self.dtype).to(self.device)

        # set iteration
        self.i = 0

        # convert to as the parameters in a model
        self.V_c_row = torch.nn.Parameter(self.V_c_row, requires_grad=False)
        self.V_c_col = torch.nn.Parameter(self.V_c_col, requires_grad=False)

        self.algo = "bp_sf"
        self.num_max_iter = self.max_iter

        sf_cfg = decoder_cfg.get("sf", decoder_cfg)
        self.w_min = int(sf_cfg.get("w_min", 0))
        self.w_max = int(sf_cfg.get("w_max", 0))
        self.n_sample = int(sf_cfg.get("n_sample", 0))
        self.topk = int(sf_cfg.get("topk", 0))
        if self.w_max < self.w_min:
            logger.warning(
                f"Invalid SF weights w_min=<{self.w_min}> w_max=<{self.w_max}>, disabling SF."
            )
            self.w_max = 0
        if self.topk <= 0 and self.w_max > 0:
            logger.warning("SF enabled (w_max>0) but topk<=0; defaulting topk to 20.")
            self.topk = 20

        if self.w_max > 0:
            N = self.H_shape[1]
            if self.max_iter != N:
                logger.info(
                    f"SF enabled: overriding max_iter <{self.max_iter}> with the "
                    f"data-qubit count N=<{N}>."
                )
            self.max_iter = N
            self.num_max_iter = self.max_iter

        self.cap = RebatchSpeedup.from_cfg(None)
        self.cap_bypass = False
        self.cap_active_last = False

        logger.info(
            f"Complete. SF: w=[{self.w_min},{self.w_max}], n_sample={self.n_sample}, topk={self.topk}."
        )

    def forward(self, io_dict):
        """Decode a batch: normalized-min-sum BP, then SF post-processing on the
        samples BP left unconverged."""
        logger.info("Initializing bp_sf decoding.")

        syndrome = io_dict["synd"].to(dtype=self.dtype).to(self.device)
        llr0 = io_dict["llr0"].to(dtype=self.dtype).to(self.device)
        self.batch_size, _ = syndrome.size()
        torch.set_default_dtype(self.dtype)

        # ---- pass 1: normalized min-sum BP, tracking oscillation counts ----
        e_out, l_out, converges, num_iters, osc = self._bp_core(
            syndrome, llr0, track_osc=True
        )

        # ---- pass 2: syndrome-flipping retry on the unconverged samples ----
        if self.w_max > 0:
            self._sf_postprocess(
                syndrome, llr0, e_out, l_out, converges, num_iters, osc
            )

        logger.info("Complete.")
        io_dict.update(
            {"e_v": e_out, "iter": num_iters, "llr": l_out, "converge": converges}
        )
        return io_dict

    def _bp_core(self, syndrome, llr0, track_osc=False):
        """Run normalized-min-sum BP on a batch of syndromes.

        Identical message passing to ``bp_norm_min_sum`` (so BP-SF's core matches the
        reference's ``ms``/``ms_scaling_factor=0`` decoder). Returns
        ``(e_out, l_out, converges, num_iters, osc)`` over the real N bits; ``osc`` is
        a per-bit oscillation count (``[batch, N]``) when ``track_osc`` else ``None``.
        """
        batch_size = syndrome.size(0)
        device, dtype = self.device, self.dtype
        N = self.H_shape[1]
        N_extended = N + 1

        l_v = torch.zeros([batch_size, N_extended], dtype=dtype, device=device)
        e_v = torch.zeros([batch_size, N_extended], dtype=dtype, device=device)

        dummy_column = torch.full(
            [batch_size, 1], float("inf"), dtype=dtype, device=device
        )
        u_init = torch.cat((llr0, dummy_column), dim=1)
        e_out = torch.zeros([batch_size, N_extended], dtype=dtype, device=device)
        l_out = torch.zeros([batch_size, N_extended], dtype=dtype, device=device)
        num_iters = torch.full([batch_size], -1, device=device)
        converges = torch.full([batch_size], 0, device=device)

        message = u_init[:, self.V_c_col]

        # check-grouped (-1)^syndrome term used by the check-node update
        syndrome_neg = torch.where(syndrome == 0.0, 1.0, -1.0).to(dtype)
        self.syndrome_neg = syndrome_neg[:, self.V_c_row]
        self.batch_size = batch_size

        # oscillation tracking: prev hard decision and per-bit flip count
        if track_osc:
            prev_hard = torch.zeros(
                [batch_size, N_extended], dtype=dtype, device=device
            )
            osc = torch.zeros([batch_size, N_extended], dtype=torch.long, device=device)
        else:
            osc = None

        self.i = 0
        while self.i < self.max_iter:
            self.i += 1

            l_v_v2c = self.v2c(l_v)
            message = self.vn_update(message, l_v_v2c)
            message = self.cn_update(message)
            message_c2v = self.c2v(message)
            l_v = self.llr_update(u_init, message_c2v)
            e_v = self.hard_decision(l_v)
            s_est = self.syndrome_estimation(e_v)

            if track_osc:
                osc += (e_v != prev_hard).to(torch.long)
                prev_hard = e_v

            indices = torch.all(s_est == syndrome, 1).nonzero()
            checker = torch.where(num_iters == -1.0)[0]
            indices = indices[torch.isin(indices, checker)]
            if indices.size()[0] > 0:
                num_iters[indices] = self.i
                e_out[indices] = e_v[indices]
                l_out[indices] = l_v[indices]
                converges[indices] = 1

            if checker.size()[0] == 0:
                break

        checker = torch.where(num_iters == -1)[0]
        e_out[checker] = e_v[checker]
        l_out[checker] = l_v[checker]
        num_iters[checker] = self.i

        osc_out = osc[:, :N] if track_osc else None
        return e_out[:, :N], l_out[:, :N], converges, num_iters, osc_out

    def _sf_postprocess(self, syndrome, llr0, e_out, l_out, converges, num_iters, osc):
        """Syndrome-flipping retry on the samples pass-1 BP left unconverged.

        For each unconverged sample, the ``topk`` most-oscillating bits are the flip
        candidates. Weight-``w`` (``w_min..w_max``) combinations of those candidates are
        tried in ascending weight: the candidate columns are XORed into the syndrome,
        BP is re-run, and on convergence the candidate bits are flipped back in the
        estimate. The lowest-indexed (== weight-then-sample ordered) converging trial
        wins, matching the reference's first-break.

        Fully vectorized: every ``(unconverged sample x candidate combo)`` trial is
        stacked into a single ``_bp_core`` call rather than looping combo-by-combo.
        BP message passing is per-sample independent, so a trial's output is identical
        whether run alone or inside the wide batch -- the result is bit-for-bit the same
        as the loop, trading the early-exit for one parallel pass (larger peak memory:
        ``U*C`` rows, where ``U`` = unconverged count, ``C`` = combo count).
        Results are written in place into ``e_out``/``l_out``/``converges``.
        """
        unconv = torch.where(converges == 0)[0]
        U = unconv.numel()
        if U == 0:
            return

        N = self.H_shape[1]
        topk = min(self.topk, N)

        # per-sample candidate bit indices, highest oscillation first: [U, topk]
        cand = torch.argsort(osc[unconv], dim=1, descending=True)[:, :topk]

        combos = [
            torch.tensor(list(slots), dtype=torch.long, device=self.device)
            for w in range(self.w_min, self.w_max + 1)
            for slots in sample_n_choose_k(range(topk), w, self.n_sample)
        ]
        C = len(combos)
        if C == 0:
            return
        # pad the ragged slot rows to a rectangle (sentinel column `topk`), then scatter
        # membership in one shot -- no per-element Python loop.
        padded = torch.nn.utils.rnn.pad_sequence(
            combos, batch_first=True, padding_value=topk
        )  # [C, Lmax], missing slots point at the dummy column
        combo_mask = torch.zeros([C, topk + 1], dtype=self.dtype, device=self.device)
        if padded.numel() > 0:
            combo_mask.scatter_(1, padded, 1.0)
        combo_mask = combo_mask[:, :topk]  # drop the sentinel column

        # syndrome shift per (sample, combo) = parity of the selected columns: [U, C, M]
        Hcols = self.H_dense[:, cand]  # [M, U, topk]
        shift = torch.einsum("muk,ck->ucm", Hcols, combo_mask).remainder(2.0)
        new_synd = (syndrome[unconv].unsqueeze(1) + shift).remainder(2.0)  # [U, C, M]
        M = new_synd.size(2)

        # one batched BP over all U*C trials
        llr_batch = llr0[unconv].unsqueeze(1).expand(U, C, N).reshape(U * C, N)
        e_r, l_r, conv_r, _, _ = self._bp_core(
            new_synd.reshape(U * C, M), llr_batch, track_osc=False
        )
        e_r = e_r.view(U, C, N)
        l_r = l_r.view(U, C, N)
        conv_r = conv_r.view(U, C)

        # pick the lowest-indexed converging combo per sample (first-break semantics)
        order = torch.arange(C, device=self.device)
        sel = (conv_r.to(torch.long) * (C - order)).argmax(dim=1)  # [U]
        has = conv_r.bool().any(dim=1)  # [U]

        u_idx = torch.arange(U, device=self.device)
        chosen_e = e_r[u_idx, sel]  # [U, N]
        chosen_l = l_r[u_idx, sel]  # [U, N] (never flipped, mirrors the loop)

        # flip the winning combo's candidate bits back in the estimate
        row_flip = torch.zeros([U, N], dtype=self.dtype, device=self.device)
        row_flip.scatter_(1, cand, combo_mask[sel])
        chosen_e = torch.where(row_flip > 0, 1.0 - chosen_e, chosen_e)

        g = unconv[has]
        e_out[g] = chosen_e[has]
        l_out[g] = chosen_l[has]
        converges[g] = 1

    # BP message-passing primitives (identical to bp_norm_min_sum)
    def v2c(self, l_v):
        return l_v[:, self.V_c_col]

    def vn_update(self, b_c2v, l_v_v2c):
        if self.i == 1:
            return b_c2v
        else:
            return l_v_v2c - b_c2v

    def cn_update(self, a_v2c):
        base = torch.tensor(2.0, dtype=self.dtype)
        exponent = torch.tensor(-(self.i), dtype=self.dtype)
        beta = torch.tensor(1.0, dtype=self.dtype) - torch.pow(base, exponent)

        sign = torch.sgn(a_v2c)
        sign = torch.where(sign == 0.0, -1.0, sign)
        sign_prod = torch.prod(sign, dim=2, keepdim=True)
        Q_sign = self.syndrome_neg * sign_prod

        abs_a_v2c = torch.abs(a_v2c)
        sorted, _ = torch.sort(abs_a_v2c, dim=2)
        min_0 = sorted[:, :, 0].unsqueeze(2)
        min_1 = sorted[:, :, 1].unsqueeze(2)
        min_result = torch.where(abs_a_v2c == min_0, min_1, min_0)

        message = beta * sign * Q_sign * min_result
        message[:, self.mask_dummy] = 0.0
        return message

    def c2v(self, b_c2v):
        data_flat = b_c2v.flatten(start_dim=1)
        partitions_flat = self.V_c_col.flatten().repeat(self.batch_size, 1)
        sum_b_c2v = torch.zeros(
            [self.batch_size, self.H_shape[1] + 1], dtype=self.dtype, device=self.device
        )
        sum_b_c2v.scatter_add_(1, partitions_flat, data_flat)
        return sum_b_c2v

    def llr_update(self, u_init, b_c2v):
        l_v = u_init + b_c2v
        l_v[:, -1] = float("inf")
        return l_v

    def hard_decision(self, l_v):
        return torch.where(l_v <= 0.0, 1.0, 0.0).to(self.dtype)

    def syndrome_estimation(self, e_v):
        temp_e = e_v
        temp_e[:, -1] = 0.0
        estimated_syndrome = temp_e[:, self.V_c_col].sum(dim=2).to(dtype=self.dtype)
        return torch.where((estimated_syndrome % 2) > 0.0, 1.0, 0.0)
