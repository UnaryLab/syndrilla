# group key -> {member knob: value the knob takes when the group key is false}
GROUP_OFF = {
    "pruning_opt": {
        "host_check_every": 0,
        "skip_converged": False,
        "compact_frac": 0,
        "osd_early_stop": False,
        "osd_prefix_scan": False,
        "osd_solve_by_pivots": False,
        "osd_skip_converged": False,
    },
    "fusion_opt": {"persistent_kernel": False, "compile": False, "fuse_vn": False},
    "mapping_opt": {
        "cn_sign_parity": False,
        "warp_per_check": False,
        "f64_int_compare": False,
        "edge_layout": "padded",
    },
    "gather_opt": {"c2v_gather": False, "vn_gather": False},
    "memory_opt": {
        "reuse_buffers": False,
        "osd_column_scan": False,
        "osd_packed_transform": False,
        "workspace_bytes": 1 << 60,
        "sparse_h": False,
    },
    "rebatch_opt": {"rebatch_opt": False},
}


def knob(cfg, key, default):
    """cfg[key] when the key is set; else the GROUP_OFF value of the first group
    that has key as a member and is set false in cfg; else default."""
    if key in cfg:
        return cfg[key]
    for group, off in GROUP_OFF.items():
        if key in off and not cfg.get(group, True):
            return off[key]
    return default
