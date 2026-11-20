"""Assembly of the quantitative /nmr response from estimator records."""
import math

from . import design as design_module
from .quadrature import summarize_quadrature

ILL_CONDITIONED_THRESHOLD = 1e3
# Two-sided 97.5 percent points of Student's t; 1.959964 is the normal limit.
_T_975 = {1: 12.706204736432095, 2: 4.302652729911275, 3: 3.1824463052837086,
          4: 2.7764451051977987, 5: 2.5705818366147395, 6: 2.4469118487916806,
          7: 2.3646242510102993, 8: 2.3060041350333784, 9: 2.2621571627409915,
          10: 2.2281388519649385, 11: 2.200985160082949, 12: 2.1788128296634177,
          13: 2.1603686564610127, 14: 2.1447866879169273, 15: 2.131449545559323,
          16: 2.1199052992210112, 17: 2.1098155778331806, 18: 2.100922040240166,
          19: 2.093024054408263, 20: 2.0859634472658626, 21: 2.079613844727662,
          22: 2.0738730679040147, 23: 2.0686576104190406, 24: 2.0638985616280205,
          25: 2.059538552753294, 26: 2.055529438642871, 27: 2.0518305164802833,
          28: 2.048407143258471, 29: 2.045229642132703, 30: 2.0422724563012373}


def _multiplier(degrees_of_freedom, known_variance):
    if known_variance:
        return 1.959963984540054
    if degrees_of_freedom is None or degrees_of_freedom <= 0:
        return None
    if degrees_of_freedom > 30:
        return 1.959963984540054
    return _T_975[degrees_of_freedom]


def _kish(values):
    total = math.fsum(values)
    if total <= 0:
        return None
    square_total = math.fsum(value * value for value in values)
    return total * total / square_total if square_total > 0 else None


def _channel_payload(index, channel, target, estimation, width):
    if index not in estimation["independent"]:
        return {"channel": channel, "status": "unidentifiable",
                "is_target": channel == target, "signal": None,
                "standard_error": None, "confidence_interval_95": None,
                "magnitude": None, "magnitude_standard_error": None,
                "note": "The design matrix cannot separate this channel from its alias class"}
    record = estimation["records"][index]
    signal, errors, intervals, magnitudes, magnitude_errors = [], [], [], [], []
    scales = set()
    degrees = set()
    known_variance = estimation["weighted"]
    for column in range(width):
        value = record["estimate"][column]
        gram_diagonal = record["gram_diagonal"][column]
        scale = record["scale"][column]
        degrees_of_freedom = record["degrees_of_freedom"][column]
        scales.add(scale)
        degrees.add(degrees_of_freedom)
        signal.append([value.real, value.imag])
        if scale is None:
            errors.append(None)
            intervals.append(None)
            magnitudes.append(abs(value))
            magnitude_errors.append(None)
            continue
        # Under circular complex noise with variance field v = E|e|^2 the
        # weighted Gram diagonal equals 2 * (component estimate variance) / v,
        # so one real/imaginary component variance is scale * gram / 2.
        component_variance = 0.5 * scale * gram_diagonal
        component_error = math.sqrt(component_variance)
        errors.append(component_error)
        multiplier = _multiplier(degrees_of_freedom, known_variance)
        if multiplier is None:
            intervals.append(None)
        else:
            half_width = multiplier * component_error
            intervals.append([[value.real - half_width, value.real + half_width],
                              [value.imag - half_width, value.imag + half_width]])
        magnitude = abs(value)
        magnitudes.append(magnitude)
        if magnitude > 0:
            # Delta method for |a|; real and imaginary components share variance.
            magnitude_errors.append(math.sqrt(max(
                (value.real ** 2 + value.imag ** 2) / (magnitude ** 2) * component_variance, 0.0)))
        else:
            magnitude_errors.append(None)
    condition = estimation["design_condition_number"]
    ill_conditioned = condition is not None and condition > ILL_CONDITIONED_THRESHOLD
    if not estimation["weighted"]:
        if scales == {None}:
            status = "identifiable_unquantified_ill_conditioned" if ill_conditioned \
                else "identifiable_unquantified"
        else:
            status = "identifiable_ill_conditioned" if ill_conditioned \
                else "identifiable_residual_variance"
    else:
        status = "identifiable_ill_conditioned" if ill_conditioned else "identifiable"
    degrees_of_freedom = next(iter(degrees)) if len(degrees) == 1 else None
    return {"channel": channel, "status": status, "is_target": channel == target,
            "signal": signal, "standard_error": errors,
            "confidence_interval_95": intervals,
            "magnitude": magnitudes, "magnitude_standard_error": magnitude_errors,
            "residual_degrees_of_freedom": degrees_of_freedom}


def _alias_payload(basis, candidates, groups):
    payload = []
    for group in groups:
        factors = []
        if len(group) > 1:
            for other in group[1:]:
                factor = design_module.alias_factor(basis, group[0], other)
                factors.append({"channel": candidates[other],
                                "phase_relation": [factor.real, factor.imag]})
        payload.append({"channels": [candidates[index] for index in group],
                        "phase_relations": factors})
    return payload


def _acquisition_summary(rows, width, all_present):
    details = []
    for row in rows:
        if row["variance"] is not None:
            profile = [row["variance"][min(column, len(row["variance"]) - 1)] for column in range(width)]
            kish = _kish([1.0 / value for value in profile])
        else:
            kish = None
        details.append({
            "variance_status": row["variance_status"],
            "repeat_count": row["repeat_count"],
            "weighted_effective_samples": kish,
            "increment": row["increment"],
            "quadrature_component": row["component"],
        })
    if all_present:
        per_column = []
        for column in range(width):
            weights = [1.0 / row["variance"][min(column, len(row["variance"]) - 1)] for row in rows]
            per_column.append(_kish(weights))
        effective = {"method": "kish_effective_sample_size", "per_time_sample": per_column,
                     "minimum": min(per_column), "maximum": max(per_column)}
    else:
        effective = {"method": "independent_acquisition_count",
                     "value": len(rows),
                     "note": "No uncertainty is available for rows without variance information"}
    return details, effective


def assemble(prepared, estimation):
    candidates = prepared["candidates"]
    width = prepared["width"]
    channels = [_channel_payload(index, candidates[index], prepared["channel"], estimation, width)
                for index in range(len(candidates))]
    target = next(channel for channel in channels if channel["is_target"])
    structure = estimation["structure"]
    acquisition_details, effective = _acquisition_summary(
        prepared["rows"], width, estimation["weighted"])
    plan = prepared["plan"]
    return {
        "model": "general-linear-model-coherence-v1",
        "receiver_convention": "observed = physical * exp(-i * receiver_phase)",
        "target_channel": prepared["channel"],
        "candidate_channels": {"source": prepared["candidate_source"],
                               "target_appended": prepared["target_added_to_candidates"],
                               "channels": channels},
        "target": target,
        "identifiability": {
            "rank": structure["rank"],
            "candidate_count": len(candidates),
            "full_rank": structure["rank"] == len(candidates),
            "condition_number": structure["condition_number"],
            "ill_conditioned": (structure["condition_number"] is not None
                                and structure["condition_number"] > ILL_CONDITIONED_THRESHOLD),
            "alias_classes": _alias_payload(prepared["basis"], candidates,
                                            structure["alias_groups"]),
            "generic_rank_deficiency": structure["generic_rank_deficiency"],
            "dependent_indices": estimation["dependent"],
        },
        "plan": {
            "cycle_size": plan["cycle_size"],
            "grid": plan["grid"],
            "quadrature": summarize_quadrature(plan["details"]),
        },
        "variance": {
            "weighted_least_squares": estimation["weighted"],
            "partial_variance_information": estimation["partial_variance"],
            "acquisitions": acquisition_details,
            "effective_acquisition": effective,
        },
        "acquisition_count": prepared["acquisition_count"],
        "total_measurements": prepared["total_measurements"],
        "alias_modulus": plan["grid"]["cycle_size"] if plan["grid"] and plan["grid"]["complete_cycle"] else None,
    }
