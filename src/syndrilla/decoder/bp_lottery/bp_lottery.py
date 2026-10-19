from bisect import bisect_left

from syndrilla.decoder.bp_lottery.pre import PrePhase

import torch
from loguru import logger

from syndrilla.decoder.bp_norm_min_sum.bp_norm_min_sum import create as _NmsPy
from syndrilla.decoder.knobs import BASE_CONFIG_KEYS, reject_unknown_keys as _reject_unknown_keys


LOTTERY_CONFIG_KEYS = BASE_CONFIG_KEYS.union({
    "flip_phase_iters", "flip_mode", "cn_pick", "cn_pick_tie_break",
    "random_machine", "vn_pick", "vn_pick_tie_break", "pre_max_lambda", "pre_max_iter",
})


STUDY_KNOBS = {
    "cn_pick": ("rand_unsat", "max_err_unsat", "all_unsat", "all"),
    "vn_pick": ("max_unsat", "random"),
}


def read_study_knobs(cfg):
    knobs = {key: cfg.get(key, values[0]) for key, values in STUDY_KNOBS.items()}
    for key, value in knobs.items():
        if value not in STUDY_KNOBS[key]:
            raise ValueError(f"{key} must be one of {STUDY_KNOBS[key]}, got {value!r}")
    knobs["cn_pick_tie_break"] = read_cn_pick_tie_break(cfg)
    return knobs


def reject_unknown_keys(cfg, extra=()):
    _reject_unknown_keys(cfg, LOTTERY_CONFIG_KEYS.union(extra), cfg.get("algorithm", "bp_lottery"))


def read_phase_knobs(cfg, extra=()):
    reject_unknown_keys(cfg, extra)
    if "flip_phase_iters" not in cfg and "flip_mode" not in cfg:
        cfg = {**cfg, "flip_phase_iters": [40], "flip_mode": ["lottery", "pre"]}
    boundaries = cfg.get("flip_phase_iters", [])
    if (not isinstance(boundaries, list) or len(boundaries) > 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in boundaries)
            or any(a > b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError("flip_phase_iters must be a nondecreasing list of up to two nonnegative ints")
    count = len(boundaries) + 1
    keys = ("cn_pick", "vn_pick", "flip_mode")
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, list) and len(value) != count:
            raise ValueError(f"{key} list must have {count} entries for {count} phases, got {len(value)}")
    phases = []
    for i in range(count):
        values = {**cfg, **{k: cfg[k][i] for k in keys if isinstance(cfg.get(k), list)}}
        phase = read_study_knobs(values)
        phase["flip_mode"] = values.get("flip_mode", "lottery")
        if phase["flip_mode"] not in ("lottery", "off", "pre"):
            raise ValueError("flip_mode must be lottery, off, or pre")
        phases.append(phase)
    return list(boundaries), phases


def set_phase(dec, i):
    if hasattr(dec, "phase_knobs"):
        index = bisect_left(dec.flip_phase_iters, i)
        if index != dec._phase_index:
            for key, value in dec.phase_knobs[index].items():
                setattr(dec, key, value)
            dec._phase_index = index


def reset_phase_state(dec):
    dec._phase_index = -1
    set_phase(dec, 1)


def vn_unsat_count(cn_val, V_c_col, N):
    """cn_val [B, M] @ H as [B, N]: per variable, the sum of cn_val over its checks.

    Scatters each edge's check value onto its variable over V_c_col; padded edges
    land on the dummy column N, which is dropped. Output dtype follows cn_val.
    """
    B, D = cn_val.shape[0], V_c_col.shape[1]
    src = cn_val.unsqueeze(2).expand(-1, -1, D).reshape(B, -1)
    return cn_val.new_zeros(B, N + 1).index_add_(1, V_c_col.reshape(-1), src)[:, :N]


def cn_row_mask(V_c_col, cn_idx, N):
    """Bool [B, N] mask of the variables on check cn_idx[b], i.e. H[cn_idx].bool()."""
    mask = torch.zeros(cn_idx.shape[0], N + 1, dtype=torch.bool, device=V_c_col.device)
    return mask.scatter_(1, V_c_col[cn_idx], True)[:, :N]


def flip_rows(l_v, idx, mask):
    """Negate l_v[b, idx[b]] in place for every row b where mask[b] is True,
    without a host sync."""
    sel = idx.unsqueeze(1)
    v = l_v.gather(1, sel)
    l_v.scatter_(1, sel, torch.where(mask.unsqueeze(1), -v, v))


def rand_rows(dec, n):
    """n uniform draws in dec.dtype from the global RNG, one per hook row: drawn
    for the whole batch of dec._B samples, then indexed by dec._hook_rows when
    the n rows are a compacted subset, so a sample's draw does not depend on
    which samples are still decoded."""
    r = torch.rand(dec._B, device=dec.device, dtype=dec.dtype)
    return r if n == dec._B else r[dec._hook_rows]


def is_flip_iter(dec, i):
    """Flip at every hook in a lottery phase, before the final iteration."""
    set_phase(dec, i)
    return getattr(dec, "flip_mode", "lottery") != "off" and 0 < i < dec.max_iter


def read_cn_pick_tie_break(cfg):
    value = cfg.get("cn_pick_tie_break", True)
    if not isinstance(value, bool):
        raise ValueError("cn_pick_tie_break must be a bool")
    return value


def read_vn_pick_tie_break(cfg):
    value = cfg.get("vn_pick_tie_break", True)
    if not isinstance(value, bool):
        raise ValueError("vn_pick_tie_break must be a bool")
    return value


def select_flip_vn(dec, unsat_cn_mask, l_v, e_v):
    batch_size = unsat_cn_mask.shape[0]
    total_unsat = unsat_cn_mask.sum(dim=1)
    N = dec.H_shape[1]
    r = dec._draw_r(batch_size)
    total_unsat_safe = total_unsat + (total_unsat == 0).float()
    unsat_cumsum = unsat_cn_mask.cumsum(dim=1)
    rand_pos = torch.floor(r * total_unsat_safe).long() + 1
    chosen_cn = (
        (unsat_cumsum >= rand_pos.unsqueeze(1)) & unsat_cn_mask
    ).float()
    chosen_cn_idx = torch.argmax(chosen_cn, dim=1)

    if dec.cn_pick == "max_err_unsat":
        count = ((e_v[:, dec.V_c_col] == 1) & ~dec.mask_dummy).sum(dim=2)
        count = count.masked_fill(~unsat_cn_mask, -1)
        chosen_cn_idx = count.argmax(dim=1)
        if dec.cn_pick_tie_break:
            tie = (count == count.max(dim=1, keepdim=True).values) & unsat_cn_mask
            pos = (r * tie.sum(dim=1)).long() + 1
            chosen_cn_idx = ((tie.cumsum(dim=1) >= pos[:, None]) & tie).long().argmax(dim=1)
    vn_unsat_counts = vn_unsat_count(unsat_cn_mask.float(), dec.V_c_col, N)
    if dec.cn_pick == "all_unsat":
        candidate_vn_mask = vn_unsat_counts > 0
    elif dec.cn_pick == "all":
        candidate_vn_mask = torch.ones_like(vn_unsat_counts, dtype=torch.bool)
    else:
        candidate_vn_mask = cn_row_mask(dec.V_c_col, chosen_cn_idx, N)

    excluded = getattr(dec, "_pick_excluded", None)
    if excluded is not None:
        candidate_vn_mask &= ~excluded[:, :N]
        dec._pick_available = candidate_vn_mask.any(1)

    nonzero = candidate_vn_mask & (l_v[:, :-1] != 0)
    candidate_vn_mask = torch.where(nonzero.any(1, keepdim=True), nonzero, candidate_vn_mask)

    llr = torch.abs(l_v[:, :-1])
    selected_vn = pick_vn(dec, candidate_vn_mask, vn_unsat_counts, llr)
    if dec.cn_pick in ("all_unsat", "all"):
        chosen_cn_idx.fill_(-1)
    return chosen_cn_idx, selected_vn


def pick_vn(dec, candidate_vn_mask, vn_unsat_counts, llr):
    """Select one node from the candidate pool."""
    batch_size = len(llr)
    count = vn_unsat_counts.masked_fill(~candidate_vn_mask, -1)
    tie = (count == count.max(dim=1, keepdim=True).values) & candidate_vn_mask
    if dec.vn_pick_tie_break:
        selected_vn = llr.masked_fill(~tie, float("inf")).argmin(dim=1)
    else:
        selected_vn = tie.float().argmax(dim=1)
    if dec.vn_pick == "random":
        r_v = (rand_rows(dec, batch_size) if dec.random_machine.lower() == "system"
               else dec.r_v[dec.i - 1, 0].repeat(batch_size))
        pos = (r_v * candidate_vn_mask.sum(dim=1)).long() + 1
        selected_vn = ((candidate_vn_mask.cumsum(dim=1) >= pos[:, None])
                       & candidate_vn_mask).long().argmax(dim=1)
    return selected_vn


def flip_with_mode(dec, syndrome, s_est, l_v):
    unsat = ((syndrome + s_est) % 2.0).bool()
    valid = unsat.any(1) & dec._active
    cn, vn = select_flip_vn(dec, unsat, l_v, getattr(dec, "_flip_e_v", None))
    if getattr(dec, "_pick_excluded", None) is not None:
        valid &= dec._pick_available
    second = torch.full_like(vn, -1)
    dec.last_flip_cn, dec.last_flip_vn, dec.last_flip_vn2 = cn, vn, second
    dec.last_flip_valid = valid
    dec.last_flip_count = valid.long()
    flip_rows(l_v, vn, valid)
    return l_v


class create(PrePhase, _NmsPy):
    """
    Lottery BP decoder: bp_norm_min_sum (eager or compiled step, row compaction)
    with the lottery sign-flip in _iter_hook during lottery phases.

    Accepts every bp_norm_min_sum key plus
        random_machine: 'sobol' (default) | 'system'   RNG for the flip pick
        cn_pick: rand_unsat (default) | max_err_unsat | all_unsat | all
            max_err_unsat selects the unsatisfied CN with the most ones in the error
            estimate; ties use lowest CN index when cn_pick_tie_break is False
            all_unsat selects from all VNs touching an unsatisfied CN
            all selects from every VN, with no syndrome filter
        cn_pick_tie_break: bool (default True), draws uniformly among max_err_unsat ties
        vn_pick: max_unsat (default) | random
            max_unsat selects the VN with the most unsatisfied CNs, ties to min |LLR|
            random selects uniformly over the candidate set
        vn_pick_tie_break: bool (default True), enables the second key; random ignores it
    """

    _config_algorithm = "bp_lottery"
    _config_keys = ()
    _base_config_keys = LOTTERY_CONFIG_KEYS

    def __init__(self, decoding_cfg, **kwargs) -> None:
        boundaries, phases = read_phase_knobs(
            {"algorithm": self._config_algorithm, **decoding_cfg}, self._config_keys)
        self.vn_pick_tie_break = read_vn_pick_tie_break(decoding_cfg)
        super().__init__(decoding_cfg, **kwargs)
        self.flip_phase_iters, self.phase_knobs, self._phase_index = boundaries, phases, -1
        set_phase(self, 1)
        self.random_machine = decoding_cfg.get("random_machine", "sobol")
        if self.random_machine.lower() not in {"sobol", "system"}:
            logger.warning(
                f"Invalid input machine type <{self.random_machine}>, default to <sobol>."
            )
            self.random_machine = "sobol"

        self._init_pre_phase(decoding_cfg)
        self.algo = "bp_lottery"

    def forward(self, io_dict):
        """bp_norm_min_sum decoding with the lottery sign-flip in _iter_hook.
        Builds the Sobol sequence (one value per iteration) first."""
        self._B = io_dict["synd"].shape[0]
        reset_phase_state(self)
        if self.random_machine.lower() == "sobol":
            sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=False)
            self.r = (
                sobol.draw(self.max_iter, dtype=torch.float32)
                .to(self.device)
                .to(self.dtype)
            )
            if any(p["vn_pick"] == "random" for p in self.phase_knobs):
                # A separate stream keeps check draws paired across VN-pick arms.
                neighbors = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
                self.r_v = neighbors.draw(self.max_iter, dtype=torch.float32)[:, 1:].to(
                    self.device, self.dtype
                )
        return self._pre_main(io_dict, super().forward)

    def _native_iter_hook(self, i, l_v, e_v, active, syndrome) -> None:
        """Lottery sign-flip at the end of each flip iteration i (is_flip_iter)
        on the unconverged rows; the next iteration reads the flipped l_v."""
        self.i = i
        if not is_flip_iter(self, i):
            return
        self._active = active
        self._flip_e_v = e_v
        self.sign_flip_cn_rand_new(syndrome, self.syndrome_estimation(e_v), l_v)

    def _draw_r(self, n):
        """[n] uniform draws for the flip pick, one per hook row."""
        if self.random_machine.lower() == "system":
            return rand_rows(self, n)
        return self.r[(self.i - 1)].repeat(n)  # sobol

    def sign_flip_cn_rand_new(self, syndrome, s_est, l_v):
        return flip_with_mode(self, syndrome, s_est, l_v)
