import math
from .cell_images import prepare, reconstruct
from .molecular_continuity import validate_bonds
from .triclinic import prepare_extended, reconstruct_extended
from .window_statistics import build_windows, grouped_statistics, time_sampling


def process(payload):
    if "box" in payload:
        return _process_legacy(payload)
    return _process_extended(payload)


def _process_legacy(payload):
    box, atoms, frames, max_step, lag = prepare(payload)
    continuous = reconstruct(box, frames, max_step)
    validate_bonds(payload, atoms, continuous)
    windows = []
    for start in range(max(0, len(continuous) - lag)):
        finish = start + lag
        squared = [math.fsum((b - a) ** 2 for a, b in zip(left, right))
                   for left, right in zip(continuous[start], continuous[finish])]
        windows.append({"start_frame": start, "end_frame": finish,
                        "squared_displacements": squared,
                        "mean_squared_displacement": math.fsum(squared) / len(atoms)})
    return {"model": "fixed-orthorhombic-cell-v1", "atoms": list(atoms),
            "continuous_positions": continuous,
            "image_mode": "explicit" if max_step is None else "bounded_step_assumption",
            "maximum_step_assumption": max_step, "lag_frames": lag, "windows": windows}


def _process_extended(payload):
    config = prepare_extended(payload)
    reconstruction = reconstruct_extended(config)
    windows, pairs, unmatched = build_windows(config, reconstruction)
    statistics = grouped_statistics(config, windows, pairs)
    return {
        "model": "triclinic-variable-cell-v1",
        "atoms": list(config["atoms"]),
        "groups": [{"id": group_id, "atoms": list(members)}
                   for group_id, members in config["groups"]],
        "times": list(config["times"]),
        "cells": [list(list(row) for row in cell) for cell in config["cells"]],
        "time_sampling": time_sampling(config["times"]),
        "image_mode": config["image_mode"],
        "reference": config["reference"],
        "maximum_step_assumption": config["max_step"],
        "lag_time": config["lag_time"],
        "lag_tolerance": config["lag_tolerance"],
        "lag_matching": {
            "matched_windows": len(pairs),
            "unmatched_frame_starts": unmatched,
        },
        "continuous_positions": [
            [list(position) if position is not None else None
             for position in frame]
            for frame in reconstruction["continuous"]
        ],
        "segment_ids": reconstruction["segments"],
        "windows": windows,
        "group_statistics": statistics,
    }
