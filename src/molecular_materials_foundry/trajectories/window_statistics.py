"""Lag matching, displacement sampling and grouped statistics."""
import bisect
import math
from .triclinic import matvec, matrix_norm_inf, ANOMALY_LIST_CAP
from ..errors import DomainError


def match_lag_pairs(times, lag_time, tolerance):
    """Pair frame indices whose elapsed time matches the requested physical lag.

    Each start frame matches at most one end frame: the nearest time.  An exact
    tie is rejected because the lag is not identifiable.
    """
    pairs = []
    unmatched = []
    for start, start_time in enumerate(times):
        target = start_time + lag_time
        insertion = bisect.bisect_left(times, target)
        candidates = []
        for index in (insertion - 2, insertion - 1, insertion, insertion + 1):
            if 0 <= index < len(times) and index > start:
                if abs(times[index] - target) <= tolerance:
                    candidates.append(index)
        candidates = sorted(set(candidates),
                            key=lambda index: (abs(times[index] - target), index))
        if not candidates:
            unmatched.append(start)
            continue
        gap = abs(abs(times[candidates[0]] - target)
                  - abs(times[candidates[1]] - target)) if len(candidates) > 1 else 1.0
        if len(candidates) > 1 and gap <= 1e-12 * max(1.0, abs(target)):
            raise DomainError("ambiguous_lag_match",
                              "Two frames match the requested lag time equally")
        end = candidates[0]
        pairs.append((start, end))
    return pairs, unmatched


def _sample_validity(config, reconstruction, start, end, atom):
    """Return (valid, reason) for one displacement sample."""
    frames = config["frames"]
    if not frames[start]["observed"] or not frames[end]["observed"]:
        return False, "missing_frame"
    # Any missing frame inside the interval contaminates the sample.
    for middle in range(start + 1, end):
        if not frames[middle]["observed"]:
            return False, "missing_frame"
    if config["image_mode"] == "explicit":
        for frame_index in range(start, end + 1):
            if (frame_index, atom) in reconstruction["conflict_frames"]:
                return False, "bond_conflict"
        return True, None
    if reconstruction["segments"][start][atom] != reconstruction["segments"][end][atom]:
        reason = reconstruction["break_reasons"].get((end, atom))
        if reason is None:
            reason = next((reconstruction["break_reasons"].get((f, atom))
                           for f in range(start + 1, end + 1)
                           if reconstruction["break_reasons"].get((f, atom))),
                          "unidentifiable")
        return False, reason
    return True, None


def displacement_vectors(config, reconstruction, start, end):
    """Laboratory and affine-removed cartesian displacement per atom."""
    cells = config["cells"]
    laboratory, affine_removed = [], []
    for atom in range(len(config["atoms"])):
        x0 = reconstruction["continuous"][start][atom]
        x1 = reconstruction["continuous"][end][atom]
        if x0 is None or x1 is None:
            laboratory.append(None)
            affine_removed.append(None)
            continue
        lab = tuple(b - a for a, b in zip(x0, x1))
        laboratory.append(lab)
        # Fractional change is invariant to the chosen unwrapping reference;
        # the start cell maps the same fractional change to the reference frame,
        # removing the affine deformation H1 H0^-1 - I of the reference cell.
        if config["image_mode"] == "explicit":
            fractional_start = matvec(config["cell_inverses"][start], x0)
            fractional_end = matvec(config["cell_inverses"][end], x1)
        else:
            fractional_start = reconstruction["continuous_fractional"][start][atom]
            fractional_end = reconstruction["continuous_fractional"][end][atom]
        if fractional_start is None or fractional_end is None:
            affine_removed.append(None)
        else:
            fractional_change = tuple(b - a for a, b
                                      in zip(fractional_start, fractional_end))
            affine_removed.append(matvec(cells[start], fractional_change))
    return laboratory, affine_removed


def build_windows(config, reconstruction):
    pairs, unmatched = match_lag_pairs(config["times"], config["lag_time"],
                                       config["lag_tolerance"])
    windows = []
    for start, end in pairs:
        validity = [_sample_validity(config, reconstruction, start, end, atom)
                    for atom in range(len(config["atoms"]))]
        laboratory, affine_removed = displacement_vectors(config, reconstruction,
                                                          start, end)
        chosen = laboratory if config["reference"] == "laboratory" else affine_removed
        squared = []
        for atom, (valid, _reason) in enumerate(validity):
            if valid:
                squared.append(math.fsum(value * value for value in chosen[atom]))
            else:
                squared.append(None)
        valid_values = [value for value in squared if value is not None]
        windows.append({
            "start_frame": start,
            "end_frame": end,
            "start_time": config["times"][start],
            "end_time": config["times"][end],
            "elapsed_time": config["times"][end] - config["times"][start],
            "sample_status": [{"valid": valid, "reason": reason}
                              for valid, reason in validity],
            "squared_displacements": squared,
            "mean_squared_displacement": (math.fsum(valid_values) / len(valid_values)
                                          if valid_values else None),
        })
    return windows, pairs, unmatched


def quantile(sorted_values, fraction):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _independent_windows(window_indices, pairs):
    """Greedy set of non-overlapping lag windows for effective sample size."""
    ordered = sorted(window_indices, key=lambda i: (pairs[i][1], pairs[i][0]))
    chosen = []
    last_end = -1
    for index in ordered:
        start, end = pairs[index]
        if start >= last_end:
            chosen.append(index)
            last_end = end
    return chosen


def _envelope_bound(config, start, end):
    """Conservative upper MSD bound given the declared per-step motion bound."""
    steps = end - start
    if config["reference"] == "laboratory":
        return steps * (config["max_step"] * config["max_step"])
    # Affine removal rescales fractional steps; bound the cartesian norm of the
    # reference cell applied to fractional motion using its norm/inverse bounds.
    norm_inf = max(matrix_norm_inf(config["cells"][f]) for f in range(start, end))
    inverse_inf = max(matrix_norm_inf(config["cell_inverses"][f])
                      for f in range(start, end))
    factor = math.sqrt(3.0) * norm_inf * (math.sqrt(3.0) * inverse_inf)
    return steps * ((factor * config["max_step"]) ** 2)


def grouped_statistics(config, windows, pairs):
    atom_index = {name: i for i, name in enumerate(config["atoms"])}
    group_results = []
    for group_id, members in config["groups"]:
        member_indices = [atom_index[name] for name in members]
        pool = []
        candidate_count = 0
        identified_count = 0
        group_windows_with_sample = set()
        anomalies = []
        envelope_values = []
        for window_index, window in enumerate(windows):
            start, end = pairs[window_index]
            window_had_sample = False
            window_had_failure = False
            reason = None
            for atom in member_indices:
                candidate_count += 1
                status = window["sample_status"][atom]
                if status["valid"]:
                    identified_count += 1
                    pool.append(window["squared_displacements"][atom])
                    window_had_sample = True
                else:
                    window_had_failure = True
                    reason = status["reason"]
                    envelope = _envelope_bound(config, start, end)
                    envelope_values.append(envelope)
            if window_had_sample:
                group_windows_with_sample.add(window_index)
            if window_had_failure and len(anomalies) < ANOMALY_LIST_CAP:
                anomalies.append({
                    "start_frame": start, "end_frame": end,
                    "start_time": window["start_time"], "end_time": window["end_time"],
                    "reason": reason,
                    "atoms_affected": [config["atoms"][atom] for atom in member_indices
                                       if not window["sample_status"][atom]["valid"]],
                })
        independent_with_sample = _independent_windows(group_windows_with_sample, pairs)
        effective_sample_size = len(independent_with_sample) * len(member_indices)
        missing_rate = (1.0 - identified_count / candidate_count) if candidate_count else None
        if pool:
            ordered = sorted(pool)
            mean = math.fsum(ordered) / len(ordered)
            variance = math.fsum((value - mean) ** 2 for value in ordered) / len(ordered)
            sem95 = 1.96 * math.sqrt(variance / len(ordered))
            statistical_interval = [max(0.0, mean - sem95), mean + sem95]
            conservative_interval = [ordered[0], max(ordered[-1], max(envelope_values))] \
                if envelope_values else [ordered[0], ordered[-1]]
            msd = mean
        else:
            statistical_interval = None
            conservative_interval = [0.0, max(envelope_values)] if envelope_values else None
            msd = None
        group_results.append({
            "group": group_id,
            "atoms": list(members),
            "candidate_samples": candidate_count,
            "identified_samples": identified_count,
            "effective_sample_size": effective_sample_size,
            "independent_window_count": len(independent_with_sample),
            "missing_rate": missing_rate,
            "mean_squared_displacement": msd,
            "msd_interval_95pct_normal": statistical_interval,
            "msd_conservative_envelope": conservative_interval,
            "distribution": _distribution(pool),
            "anomalous_intervals_count": candidate_count - identified_count,
            "anomalous_intervals_truncated": (candidate_count - identified_count) > len(anomalies),
            "anomalous_intervals": anomalies,
        })
    return group_results


def _distribution(pool):
    if not pool:
        return {"count": 0, "minimum": None, "quartile_1": None, "median": None,
                "quartile_3": None, "maximum": None, "mean": None}
    ordered = sorted(pool)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "quartile_1": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.5),
        "quartile_3": quantile(ordered, 0.75),
        "maximum": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
    }


def time_sampling(times):
    deltas = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    if not deltas:
        return {"uniform": True, "minimum_delta": None, "maximum_delta": None}
    uniform = all(math.isclose(delta, deltas[0], rel_tol=1e-9, abs_tol=0.0)
                  for delta in deltas)
    return {"uniform": uniform, "minimum_delta": min(deltas),
            "maximum_delta": max(deltas)}
