"""Combine observations of one symmetry orbit within a single batch.

Negative background-corrected intensities are physically valid and never
rejected here.  Observations whose reported variance is not a positive number
cannot enter variance-weighted statistics; they are flagged rather than
silently dropped.
"""
import math
from .batch_input import orbit_representative


def weighted_estimate(observations):
    """Variance-weighted mean with diagnostics for each contribution."""
    valid = [entry for entry in observations if entry["variance_valid"]]
    if not valid:
        return None
    smallest = min(entry["variance"] for entry in valid)
    weights = [smallest / entry["variance"] for entry in valid]
    total_weight = math.fsum(weights)
    mean = math.fsum(entry["intensity"] * weight
                     for entry, weight in zip(valid, weights)) / total_weight
    variance = smallest / total_weight
    chi_squared = math.fsum((entry["intensity"] - mean) ** 2 / entry["variance"]
                            for entry in valid)
    contributions = []
    sigma = math.sqrt(variance)
    for entry, weight in zip(valid, weights):
        residual = entry["intensity"] - mean
        z = residual / math.sqrt(entry["variance"])
        # Leave-one-out shift of the mean, expressed in merged standard errors.
        weight_fraction = weight / total_weight
        if len(valid) > 1:
            leave_one_out = mean - weight_fraction * residual / (1.0 - weight_fraction)
            shift_in_sigma = abs(mean - leave_one_out) / sigma if sigma > 0.0 else 0.0
        else:
            shift_in_sigma = None
        contributions.append({
            "observation": entry,
            "weight_fraction": weight / total_weight,
            "standardized_residual": z,
            "leave_one_out_shift_sigma": shift_in_sigma,
            "outlier": abs(z) > 3.0,
        })
    return {
        "mean": mean,
        "variance": variance,
        "chi_squared": chi_squared,
        "degrees_of_freedom": len(valid) - 1,
        "contributions": contributions,
        "n_valid": len(valid),
    }


def build_batch_classes(batch, rotations, friedel, scale_only=False):
    """Assign every observation to a symmetry/Friedel orbit in the reference frame.

    With ``scale_only`` and Friedel opposites kept separate, Friedel-averaged
    observations are dropped because their averaged value is not comparable to
    a single opposite when an anomalous signal is present.
    """
    grouped = {}
    for entry in batch["observations"]:
        if scale_only and entry["anomalous"] == "merge":
            continue
        key = orbit_representative(entry["hkl_reference"], rotations, friedel)
        grouped.setdefault(key, []).append(entry)
    classes = {}
    for key, members in grouped.items():
        estimate = weighted_estimate(members)
        classes[key] = {
            "key": key,
            "batch_index": batch["index"],
            "batch_id": batch["id"],
            "members": members,
            "estimate": estimate,
        }
    return classes
