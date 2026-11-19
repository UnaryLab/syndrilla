# group key -> {member knob: value the knob takes when the group key is false}
GROUP_OFF = {
    "pruning_opt": {
        "host_check_every": 0,
        "skip_converged": False,
        "compact_frac": 0,
        "early_stop": False,
        "prefix_scan": False,
        "solve_by_pivots": False,
    },
    "fusion_opt": {"persistent_kernel": False, "compile": False, "fuse_vn": False,
                   "fused": False},
    "mapping_opt": {
        "cn_sign_parity": False,
        "warp_per_check": False,
        "f64_int_compare": False,
        "edge_layout": "padded",
    },
    "gather_opt": {"c2v_gather": False, "vn_gather": False},
    "memory_opt": {
        "reuse_buffers": False,
        "column_scan": False,
        "packed_transform": False,
        "workspace_bytes": 1 << 60,
        "sparse_h": False,
    },
    "rebatch_opt": {"rebatch_opt": False},
}


SHARED_CONFIG_KEYS = frozenset({
    "algorithm", "check_type", "dtype", "device", "force_pytorch", "rebatch_opt_params",
}).union(GROUP_OFF)

OSD_ONLY = frozenset({
    "early_stop", "prefix_scan", "solve_by_pivots", "column_scan",
    "packed_transform", "fused",
})

BASE_CONFIG_KEYS = SHARED_CONFIG_KEYS.union(
    {"early_termination", "max_iter"}, GROUP_OFF,
    {key for members in GROUP_OFF.values() for key in members
     if key not in OSD_ONLY and key not in {"workspace_bytes", "sparse_h"}},
)


def reject_unknown_keys(cfg, legal, algo):
    for key in cfg:
        if key not in legal:
            raise ValueError(f"unsupported key {key} for {algo}")


def knob(cfg, key, default):
    """cfg[key] when the key is set; else the GROUP_OFF value of the first group
    that has key as a member and is set false in cfg; else default."""
    if key in cfg:
        return cfg[key]
    for group, off in GROUP_OFF.items():
        if key in off and not cfg.get(group, True):
            return off[key]
    return default
