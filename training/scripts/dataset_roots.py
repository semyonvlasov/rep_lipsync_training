#!/usr/bin/env python3

"""
Helpers for collecting configured processed dataset roots.
"""


def get_dataset_roots(cfg):
    data_cfg = cfg.get("data", {})
    roots = [
        data_cfg.get("voxceleb2_root"),
        data_cfg.get("lrs2_root"),
        data_cfg.get("hdtf_root"),
        data_cfg.get("talkvid_root"),
    ]
    return [root for root in roots if root]
