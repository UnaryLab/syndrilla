import itertools
import math
import random

import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as _Base


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


class create(_Base):
    """BP-SF decoder (normalized min-sum BP + syndrome-flipping post-processing).

    The belief-propagation core is the ``bp_norm_min_sum`` forward (parallel
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

    def __init__(self, decoding_cfg, **kwargs) -> None:
        super(create, self).__init__(decoding_cfg, **kwargs)

        logger.info("Creating bp_sf decoder.")

        self._init_sf(decoding_cfg)
        if self.compile:
            # beta of every iteration, same formula as bp_norm_min_sum
            i = torch.arange(1, self.max_iter + 1).to(self.dtype)
            self.betas = (
                torch.tensor(1.0, dtype=self.dtype)
                - torch.pow(torch.tensor(2.0, dtype=self.dtype), -i)
            ).to(self.device)

        self.algo = "bp_sf"
        self.cap = None
        # oscillation counts of the main pass, None outside it
        self._osc = None

        logger.info(
            f"Complete. SF: w=[{self.w_min},{self.w_max}], n_sample={self.n_sample}, topk={self.topk}."
        )

    def _init_sf(self, decoding_cfg):
        """Parse the SF parameters (w_min, w_max, n_sample, topk), set max_iter and
        num_max_iter to N when SF is on, and build V_v_row. Uses only H_shape,
        V_c_row, V_c_col, device and max_iter, which bp_norm_min_sum and
        bp_norm_min_sum_cuda both set."""
        M, N = self.H_shape
        sf_cfg = decoding_cfg.get("sf", decoding_cfg)
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
            if self.max_iter != N:
                logger.info(
                    f"SF enabled: overriding max_iter <{self.max_iter}> with the "
                    f"data-qubit count N=<{N}>."
                )
            self.max_iter = self.num_max_iter = N

        # padded CSC [N, max column degree]: the checks on each variable, padded with
        # the dummy check M; used to compute the syndrome shift of a candidate flip
        # set during SF post-processing.
        edge = self.V_c_col != N
        rows, cols = self.V_c_row[edge], self.V_c_col[edge]
        order = torch.argsort(cols, stable=True)
        rows, cols = rows[order], cols[order]
        counts = torch.bincount(cols, minlength=N)
        start = torch.cumsum(counts, 0) - counts
        self.V_v_row = torch.full([N, int(counts.max())], M, dtype=torch.long, device=self.device)
        self.V_v_row[cols, torch.arange(cols.numel(), device=self.device) - start[cols]] = rows

    def _iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """In the main pass, per bit, count how often the hard decision differs
        from the previous iteration, starting from an all-zero prior."""
        if self._osc is None:
            return
        rows = self._hook_rows
        self._osc[rows] += e_v != self._prev_hard[rows]
        self._prev_hard[rows] = e_v

    def forward(self, io_dict):
        """Decode a batch: the bp_norm_min_sum forward with oscillation counts, then
        SF post-processing on the samples BP left unconverged."""
        logger.info("Initializing bp_sf decoding.")

        syndrome = io_dict["synd"].to(dtype=self.dtype).to(self.device)
        llr0 = io_dict["llr0"].to(dtype=self.dtype).to(self.device)
        B = syndrome.shape[0]
        N_extended = self.H_shape[1] + 1
        self._osc = torch.zeros([B, N_extended], dtype=torch.long, device=self.device)
        self._prev_hard = torch.zeros([B, N_extended], dtype=self.dtype, device=self.device)
        super(create, self).forward(io_dict)
        osc = self._osc[:, :-1]
        self._osc = self._prev_hard = None

        if self.w_max > 0:
            self._sf_postprocess(
                syndrome,
                llr0,
                io_dict["e_v"],
                io_dict["llr"],
                io_dict["converge"],
                io_dict["iter"],
                osc,
            )

        logger.info("Complete.")
        return io_dict

    def _bp_core(self, syndrome, llr0):
        """SF retry decode through the bp_norm_min_sum forward. Returns
        ``(e_out, l_out, converges, num_iters, None)`` over the real N bits."""
        out = super(create, self).forward({"synd": syndrome, "llr0": llr0})
        return out["e_v"], out["llr"], out["converge"], out["iter"], None

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
        M = self.H_shape[0]
        Hcols = torch.zeros([U, topk, M + 1], dtype=self.dtype, device=self.device)
        Hcols.scatter_(2, self.V_v_row[cand], 1.0)
        Hcols = Hcols[:, :, :M].permute(2, 0, 1)  # H[:, cand]: [M, U, topk]
        shift = torch.einsum("muk,ck->ucm", Hcols, combo_mask).remainder(2.0)
        new_synd = (syndrome[unconv].unsqueeze(1) + shift).remainder(2.0)  # [U, C, M]
        M = new_synd.size(2)

        # one batched BP over all U*C trials
        llr_batch = llr0[unconv].unsqueeze(1).expand(U, C, N).reshape(U * C, N)
        e_r, l_r, conv_r, _, _ = self._bp_core(new_synd.reshape(U * C, M), llr_batch)
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
