"""Multi-batch reflection merging with overlap-based relative scaling."""
import math
from .batch_input import prepare, orbit_representative
from .affine_group import (
    common_reference_group,
    extinction_reason,
    translation_as_text,
)
from .class_merging import build_batch_classes
from .relative_scaling import estimate_scales
from .statistics import chi_survival


def _rotations(operations):
    return [operation[0] for operation in operations]


def measured_orbit_representatives(measured_reference, rotations, friedel):
    return {orbit_representative(hkl, rotations, friedel) for hkl in measured_reference}


def _merge_scaled(estimates):
    """Combine scaled batch means with inverse-variance weights."""
    weights = []
    for mean, variance in estimates:
        weight = 1.0 / variance
        weights.append(weight)
    total = math.fsum(weights)
    intensity = math.fsum(mean * weight for (mean, _), weight in zip(estimates, weights)) / total
    variance = 1.0 / total
    chi_squared = math.fsum((mean - intensity) ** 2 / variance
                            for (mean, variance) in estimates)
    return intensity, variance, chi_squared


def _build_contribution(item, inverse_scale, friedel):
    """Assemble one batch's contribution to a reflection, with per-observation flags."""
    any_outlier = False
    contribution = {
        "batch": None,
        "n_observations": len(item["members"]),
        "observations": [],
    }
    for member in item["members"]:
        excluded = ((not friedel and member["anomalous"] == "merge")
                    or not member["variance_valid"])
        if excluded:
            contribution["observations"].append({
                "local_index": member["local_index"],
                "hkl_batch": list(member["hkl_batch"]),
                "hkl_reference": list(member["hkl_reference"]),
                "intensity": member["intensity"],
                "variance": None,
                "anomalous": member["anomalous"],
                "included": False,
            })
    estimate = item["estimate"]
    usable = False
    if estimate is not None:
        scaled_mean = inverse_scale * estimate["mean"]
        scaled_variance = inverse_scale * inverse_scale * estimate["variance"]
        # When Friedel opposites are preserved, a batch whose only valid
        # readings are Friedel-averaged cannot populate a single opposite.
        valid_members = [member for member in item["members"] if member["variance_valid"]]
        if friedel or any(member["anomalous"] != "merge" for member in valid_members):
            usable = True
        contribution["scaled_mean"] = scaled_mean
        contribution["scaled_variance"] = scaled_variance
        contribution["within_batch_chi_squared"] = estimate["chi_squared"]
        contribution["within_batch_degrees_of_freedom"] = estimate["degrees_of_freedom"]
        for detail in estimate["contributions"]:
            observation = detail["observation"]
            if not friedel and observation["anomalous"] == "merge":
                continue
            if detail["outlier"]:
                any_outlier = True
            contribution["observations"].append({
                "local_index": observation["local_index"],
                "hkl_batch": list(observation["hkl_batch"]),
                "hkl_reference": list(observation["hkl_reference"]),
                "intensity": observation["intensity"],
                "variance": observation["variance"],
                "anomalous": observation["anomalous"],
                "included": True,
                "weight_fraction": detail["weight_fraction"],
                "standardized_residual": detail["standardized_residual"],
                "leave_one_out_shift_sigma": detail["leave_one_out_shift_sigma"],
                "outlier": detail["outlier"],
            })
    contribution["batch"] = item["batch_id"]
    return contribution, any_outlier, usable, (
        (contribution["scaled_mean"], contribution["scaled_variance"])
        if estimate is not None and usable else None)


def process(payload):
    batches, friedel, expected = prepare(payload)
    basis_changes = [batch["basis_change"] for batch in batches]
    batch_groups = [batch["group"] for batch in batches]
    operations = common_reference_group(batch_groups, basis_changes)
    rotations = _rotations(operations)

    # Scale classes use the request's Friedel convention.  When opposites are
    # preserved, a Friedel-averaged observation cannot be compared to a single
    # opposite, so such observations are excluded from the overlap scale fit.
    scale_classes = [build_batch_classes(batch, rotations, friedel,
                                         scale_only=(not friedel))
                     for batch in batches]
    scaling = estimate_scales(scale_classes)
    scales = scaling["scales"]

    batch_ids = [batch["id"] for batch in batches]
    group_of_batch = {}
    scale_group_records = []
    for group_index, group in enumerate(scaling["groups"]):
        for node in group["members"]:
            group_of_batch[node] = group_index
        anchor = group["anchor"]
        record = {
            "scale_group": group_index,
            "anchor_batch": batch_ids[anchor],
            "status": "unanchored" if len(group["members"]) == 1 else group["status"],
            "batch_scales": [
                {"batch": batch_ids[node], "scale": scales[node],
                 "anchored": node != anchor or len(group["members"]) > 1}
                for node in group["members"]
            ],
            "pair_estimates": [
                {
                    "batch_i": batch_ids[edge["i"]],
                    "batch_j": batch_ids[edge["j"]],
                    "scale_ratio_j_over_i": edge["rho"],
                    "relative_standard_error": edge["relative_standard_error"],
                    "overlapping_reflections": edge["n_shared"],
                    "reflections_used": edge["n_classes"],
                    "chi_squared": edge["chi_squared"],
                    "degrees_of_freedom": edge["degrees_of_freedom"],
                    "p_value": edge["p_value"],
                    "status": edge["status"],
                    "excluded_classes": edge.get("excluded_classes", []),
                }
                for edge in group["edges"]
            ],
        }
        if len(group["members"]) > 1:
            record["cycle_consistency"] = {
                "chi_squared": group["cycle_chi_squared"],
                "degrees_of_freedom": group["cycle_degrees_of_freedom"],
                "p_value": group["cycle_p_value"],
                "status": group["status"],
            }
        scale_group_records.append(record)

    # Re-group observations for the output convention (Friedel kept or merged).
    output_classes = [build_batch_classes(batch, rotations, friedel) for batch in batches]
    reflection_keys = sorted({key for classes in output_classes for key in classes})
    reflections = []
    excluded_observations = []
    invalid_variance_count = 0
    for batch in batches:
        for entry in batch["observations"]:
            if not entry["variance_valid"]:
                invalid_variance_count += 1
                excluded_observations.append({
                    "batch": entry["batch_id"],
                    "local_index": entry["local_index"],
                    "hkl": list(entry["hkl_reference"]),
                    "reason": "non_positive_or_invalid_variance",
                })
            elif not friedel and entry["anomalous"] == "merge":
                excluded_observations.append({
                    "batch": entry["batch_id"],
                    "local_index": entry["local_index"],
                    "hkl": list(entry["hkl_reference"]),
                    "reason": "friedel_averaged_observation_under_preserve_request",
                })

    for key in reflection_keys:
        batch_contributions = []
        flags = []
        total_observations = 0
        any_outlier = False
        # Batch means can only be combined inside one scale group; batches in
        # different groups have no supported relative scale, so they are never
        # force-merged, even when they happen to touch the same reflection.
        estimates_by_group = {}
        contributing_groups = set()
        for node, classes in enumerate(output_classes):
            item = classes.get(key)
            if item is None:
                continue
            total_observations += len(item["members"])
            node_group = group_of_batch[node]
            inverse_scale = 1.0 / scales[node]
            contribution, outlier, usable, estimate = _build_contribution(
                item, inverse_scale, friedel)
            contribution["scale_group"] = node_group
            batch_contributions.append(contribution)
            any_outlier = any_outlier or outlier
            if usable:
                estimates_by_group.setdefault(node_group, []).append(estimate)
                contributing_groups.add(node_group)
        if any_outlier:
            flags.append("outliers_present")
        if total_observations == 1:
            flags.append("single_observation")
        if not contributing_groups:
            reflections.append({
                "hkl": list(key),
                "status": "no_valid_variance",
                "flags": flags,
                "intensity": None,
                "variance": None,
                "n_batches": len(batch_contributions),
                "n_observations": total_observations,
                "scale_groups_observing": sorted(contributing_groups | {
                    c["scale_group"] for c in batch_contributions}),
                "batch_contributions": batch_contributions,
            })
            continue
        if len(contributing_groups) > 1:
            # No supported relative scale joins these groups: report the
            # situation explicitly instead of inventing a merged number.
            flags.append("not_mergable_scale_groups")
            per_group = []
            for node_group in sorted(contributing_groups):
                mean, var, chi = _merge_scaled(estimates_by_group[node_group])
                per_group.append({
                    "scale_group": node_group,
                    "anchor_batch": scale_group_records[node_group]["anchor_batch"],
                    "intensity": mean,
                    "variance": var,
                    "standard_uncertainty": math.sqrt(var),
                    "n_batch_estimates": len(estimates_by_group[node_group]),
                    "chi_squared": chi,
                })
            reflections.append({
                "hkl": list(key),
                "status": "scale_groups_not_comparable",
                "flags": flags,
                "intensity": None,
                "variance": None,
                "n_batches": len(batch_contributions),
                "n_observations": total_observations,
                "scale_groups_observing": sorted(contributing_groups),
                "per_scale_group_intensity": per_group,
                "batch_contributions": batch_contributions,
            })
            continue
        group_index = next(iter(contributing_groups))
        if scale_group_records[group_index]["status"] == "tension":
            flags.append("scale_tension")
        scaled_estimates = estimates_by_group[group_index]
        intensity, variance, chi_squared = _merge_scaled(scaled_estimates)
        degrees_of_freedom = len(scaled_estimates) - 1
        p_value = chi_survival(chi_squared, degrees_of_freedom)
        status = "negative_intensity" if intensity < 0.0 else "ok"
        if p_value is not None and p_value < 0.05:
            flags.append("between_batch_tension")
        reflections.append({
            "hkl": list(key),
            "scale_group": group_index,
            "status": status,
            "flags": flags,
            "intensity": intensity,
            "variance": variance,
            "standard_uncertainty": math.sqrt(variance),
            "chi_squared": chi_squared,
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": p_value,
            "n_batches": len(batch_contributions),
            "n_observations": total_observations,
            "batch_contributions": batch_contributions,
        })

    if not friedel:
        indexed = {tuple(item["hkl"]): item for item in reflections}
        for item in reflections:
            if "friedel_partner" in item:
                continue
            opposite = orbit_representative(tuple(-x for x in item["hkl"]), rotations, False)
            partner = indexed.get(tuple(opposite))
            if partner is None or item["intensity"] is None or partner["intensity"] is None:
                continue
            delta = item["intensity"] - partner["intensity"]
            delta_variance = item["variance"] + partner["variance"]
            pair = {"partner_hkl": list(opposite), "friedel_delta": delta,
                    "friedel_delta_variance": delta_variance,
                    "friedel_delta_sigma": math.sqrt(delta_variance)}
            item["friedel_contrast"] = pair
            partner["friedel_contrast"] = {
                "partner_hkl": item["hkl"], "friedel_delta": -delta,
                "friedel_delta_variance": delta_variance,
                "friedel_delta_sigma": math.sqrt(delta_variance)}

    # Systematic absences versus unmeasured reflections, in the reference frame.
    # Friedel opposites are treated as the same reflection whenever they are merged.
    preserve_friedel = friedel
    measured_reference = {}
    for batch in batches:
        for entry in batch["observations"]:
            measured_reference.setdefault(entry["hkl_reference"], []).append(entry)
    candidate_hls = set(measured_reference)
    if expected:
        candidate_hls |= set(expected)
    absence_records = {}
    unmeasured = []
    for hkl in sorted(candidate_hls):
        reason = extinction_reason(hkl, operations, preserve_friedel)
        representative = orbit_representative(hkl, rotations, friedel)
        if reason is not None:
            rotation, translation = reason
            if representative not in absence_records:
                absence_records[representative] = {
                    "hkl": list(representative),
                    "extinction_operation": {
                        "rotation": [list(row) for row in rotation],
                        "translation": translation_as_text(translation),
                    },
                    "observed": False,
                    "conflicting_observations": [],
                }
            if hkl in measured_reference:
                record = absence_records[representative]
                record["observed"] = True
                for entry in measured_reference[hkl]:
                    record["conflicting_observations"].append({
                        "batch": entry["batch_id"],
                        "local_index": entry["local_index"],
                        "hkl": list(hkl),
                        "intensity": entry["intensity"],
                    })
        elif expected and representative not in measured_orbit_representatives(
                measured_reference, rotations, friedel):
            if not any(item["hkl"] == list(representative) for item in unmeasured):
                unmeasured.append({"hkl": list(representative)})

    warnings = []
    if len(batches) > 1:
        for record in scale_group_records:
            if record["status"] == "unanchored":
                warnings.append({
                    "code": "batch_not_comparable",
                    "message": "Batch " + record["anchor_batch"]
                               + " has no supported overlap; it was not force-merged",
                })
            elif record["status"] == "tension":
                warnings.append({
                    "code": "scale_cycle_tension",
                    "message": "Overlap scales for scale group " + str(record["scale_group"])
                               + " are internally inconsistent",
                })
    if invalid_variance_count:
        warnings.append({
            "code": "invalid_variance",
            "message": str(invalid_variance_count)
                       + " observation(s) have non-positive variance and are excluded from statistics",
        })
    if scaling["scale_outliers"]:
        warnings.append({
            "code": "scale_outliers",
            "message": str(len(scaling["scale_outliers"]))
                       + " overlap contribution(s) exceeded 3 sigma and were left out of scale fits",
        })
    non_comparable = [ref for ref in reflections
                      if ref["status"] == "scale_groups_not_comparable"]
    if non_comparable:
        warnings.append({
            "code": "reflection_not_merged_across_scale_groups",
            "message": str(len(non_comparable))
                       + " reflection(s) are observed by batches without a supported relative scale;"
                         " per-scale-group intensities are reported separately",
        })
    for record in absence_records.values():
        if record["observed"]:
            warnings.append({
                "code": "extinction_conflict",
                "message": "Observed intensity at systematically extinct reflection "
                           + str(record["hkl"]),
            })
    if not friedel:
        excluded_friedel = sum(1 for item in excluded_observations
                               if item["reason"] == "friedel_averaged_observation_under_preserve_request")
        if excluded_friedel:
            warnings.append({
                "code": "friedel_averaged_observations_excluded",
                "message": str(excluded_friedel)
                           + " Friedel-averaged observation(s) cannot populate preserved opposite reflections",
            })

    return {
        "model": "explicit-affine-batch-scaling-v1",
        "friedel_merged": friedel,
        "batches": [
            {"id": batch["id"], "basis_change": [list(row) for row in batch["basis_change"]],
             "anomalous": batch["anomalous"],
             "space_group_operations": len(batch["group"]),
             "n_observations": len(batch["observations"])}
            for batch in batches
        ],
        "reference_space_group": [
            {"rotation": [list(row) for row in rotation],
             "translation": translation_as_text(translation)}
            for rotation, translation in operations
        ],
        "scale_groups": scale_group_records,
        "scale_outliers": scaling["scale_outliers"],
        "unsupported_overlaps": [
            {"batch_i": batch_ids[item["batch_i"]],
             "batch_j": batch_ids[item["batch_j"]],
             "overlapping_reflections": item["n_shared"],
             "reason": item["reason"]}
            for item in scaling["unsupported_edges"]
        ],
        "reflections": reflections,
        "systematic_absences": sorted(absence_records.values(),
                                      key=lambda item: item["hkl"]),
        "unmeasured_reflections": unmeasured,
        "excluded_observations": excluded_observations,
        "warnings": warnings,
    }
