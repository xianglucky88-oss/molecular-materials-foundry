"""Relative batch scales estimated only from overlapping observations.

Two batches are comparable when their overlap supports a weighted
least-squares ratio between their intensity scales.  Scales are propagated
over the overlap graph; batches with no supported overlap stay in their own
scale group and are never force-merged.
"""
import math
from .statistics import chi_survival, tension_status

MIN_OVERLAP_CLASSES = 2
MAX_RELATIVE_STANDARD_ERROR = 0.5
OUTLIER_Z = 3.0


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def pair_scale(x_entries, y_entries, excluded=None):
    """Weighted least squares for y ~= rho * x through the origin.

    Each entry is ``(class_key, mean, variance)``.  Keys must already be
    matched by the caller.  ``excluded`` is a set of class keys left out of
    the fit outright.  A median-ratio fit first identifies classes whose
    standardized residual exceeds three sigma; those classes are reported as
    outliers and are not used to pull the weighted least-squares estimate.
    """
    excluded = excluded or set()
    x_by_key = {entry[0]: entry for entry in x_entries if entry[0] not in excluded}
    y_by_key = {entry[0]: entry for entry in y_entries if entry[0] not in excluded}
    shared = sorted(set(x_by_key) & set(y_by_key))
    if len(shared) < MIN_OVERLAP_CLASSES:
        return {"supported": False, "reason": "fewer_than_two_overlapping_reflections",
                "n_classes": len(shared), "shared_keys": shared}
    ratios = []
    for key in shared:
        x_mean = x_by_key[key][1]
        y_mean = y_by_key[key][1]
        if x_mean > 0.0 and y_mean > 0.0:
            ratios.append(y_mean / x_mean)
    if not ratios:
        return {"supported": False, "reason": "overlap_has_no_signal",
                "n_classes": len(shared), "shared_keys": shared}
    median_rho = _median(ratios)

    def residual_records(rho):
        records = []
        for key in shared:
            _, x_mean, x_var = x_by_key[key]
            _, y_mean, y_var = y_by_key[key]
            residual = y_mean - rho * x_mean
            sigma = math.sqrt(max(y_var + rho * rho * x_var, 1e-300))
            records.append((key, residual, sigma, residual / sigma))
        return records

    # Robust outlier screen at the median ratio so a single gross outlier
    # cannot mask itself by dragging the least-squares fit toward it.
    robust_outliers = {key for key, _, _, z in residual_records(median_rho)
                       if abs(z) > OUTLIER_Z}
    fitting_keys = [key for key in shared if key not in robust_outliers]
    if len(fitting_keys) < MIN_OVERLAP_CLASSES:
        return {"supported": False, "reason": "fewer_than_two_overlapping_reflections",
                "n_classes": len(shared), "shared_keys": shared}
    rho = median_rho
    for _ in range(100):
        numerator = 0.0
        denominator = 0.0
        for key in fitting_keys:
            _, x_mean, x_var = x_by_key[key]
            _, y_mean, y_var = y_by_key[key]
            weight = 1.0 / (y_var + rho * rho * x_var)
            numerator += weight * x_mean * y_mean
            denominator += weight * x_mean * x_mean
        if denominator <= 0.0 or not math.isfinite(denominator):
            return {"supported": False, "reason": "overlap_has_no_signal",
                    "n_classes": len(shared), "shared_keys": shared}
        updated = numerator / denominator
        if abs(updated - rho) <= 1e-12 * max(1.0, abs(rho)):
            rho = updated
            break
        rho = updated
    if rho <= 0.0 or not math.isfinite(rho):
        return {"supported": False, "reason": "non_positive_scale_ratio",
                "n_classes": len(shared), "shared_keys": shared}
    information = 0.0
    chi_squared = 0.0
    for key in fitting_keys:
        _, x_mean, x_var = x_by_key[key]
        _, y_mean, y_var = y_by_key[key]
        residual_variance = y_var + rho * rho * x_var
        information += x_mean * x_mean / residual_variance
        chi_squared += (y_mean - rho * x_mean) ** 2 / residual_variance
    if information <= 0.0:
        return {"supported": False, "reason": "overlap_has_no_signal",
                "n_classes": len(shared), "shared_keys": shared}
    standard_error = 1.0 / math.sqrt(information)
    relative_standard_error = standard_error / rho
    if relative_standard_error > MAX_RELATIVE_STANDARD_ERROR:
        return {"supported": False, "reason": "scale_poorly_constrained",
                "n_classes": len(shared), "shared_keys": shared,
                "rho": rho, "relative_standard_error": relative_standard_error}
    degrees_of_freedom = len(fitting_keys) - 1
    p_value = chi_survival(chi_squared, degrees_of_freedom)
    # Diagnostics cover every shared class, including the screened outliers.
    outliers = [
        {"class": list(key), "residual": residual, "residual_sigma": sigma,
         "standardized_residual": z, "outlier": key in robust_outliers}
        for key, residual, sigma, z in residual_records(rho)
    ]
    return {
        "supported": True,
        "n_classes": len(fitting_keys),
        "n_shared": len(shared),
        "shared_keys": shared,
        "rho": rho,
        "standard_error": standard_error,
        "relative_standard_error": relative_standard_error,
        "chi_squared": chi_squared,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "status": tension_status(p_value),
        "outliers": outliers,
        "robust_outlier_keys": robust_outliers,
    }


def connected_components(node_count, edges):
    adjacency = {node: set() for node in range(node_count)}
    for edge in edges:
        adjacency[edge["i"]].add(edge["j"])
        adjacency[edge["j"]].add(edge["i"])
    seen = set()
    components = []
    for start in range(node_count):
        if start in seen:
            continue
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency[node] - seen)
        components.append(sorted(component))
    return components


def solve_scales(node_count, edges):
    """Solve log-scales from log(rho) observations on the overlap graph."""
    components = connected_components(node_count, edges)
    scales = {}
    groups = []
    for component in components:
        anchor = component[0]
        if len(component) == 1:
            scales[anchor] = 1.0
            groups.append({"members": component, "anchor": anchor, "edges": []})
            continue
        index = {node: position for position, node in enumerate(component)}
        size = len(component)
        laplacian = [[0.0] * size for _ in range(size)]
        rhs = [0.0] * size
        group_edges = []
        for edge in edges:
            if edge["i"] not in index or edge["j"] not in index:
                continue
            group_edges.append(edge)
            i, j = index[edge["i"]], index[edge["j"]]
            variance = max(edge["relative_standard_error"] ** 2, 1e-18)
            weight = 1.0 / variance
            laplacian[i][i] += weight
            laplacian[j][j] += weight
            laplacian[i][j] -= weight
            laplacian[j][i] -= weight
            observation = math.log(edge["rho"])
            rhs[i] -= weight * observation
            rhs[j] += weight * observation
        anchor_position = index[anchor]
        reduced = [row[:anchor_position] + row[anchor_position + 1:]
                   for row_index, row in enumerate(laplacian) if row_index != anchor_position]
        reduced_rhs = [value for position, value in enumerate(rhs) if position != anchor_position]
        pivot = _gaussian_elimination(reduced, reduced_rhs)
        solution = [0.0] * size
        cursor = 0
        for node in component:
            if node == anchor:
                continue
            solution[index[node]] = pivot[cursor]
            cursor += 1
        chi_squared = 0.0
        for edge in group_edges:
            predicted = solution[index[edge["j"]]] - solution[index[edge["i"]]]
            observation = math.log(edge["rho"])
            variance = max(edge["relative_standard_error"] ** 2, 1e-18)
            chi_squared += (observation - predicted) ** 2 / variance
        degrees_of_freedom = len(group_edges) - (size - 1)
        p_value = chi_survival(chi_squared, degrees_of_freedom) if degrees_of_freedom > 0 else None
        for node in component:
            position = index[node]
            if node == anchor:
                scales[node] = 1.0
            else:
                scales[node] = math.exp(solution[position])
        groups.append({"members": component, "anchor": anchor, "edges": group_edges,
                       "cycle_chi_squared": chi_squared,
                       "cycle_degrees_of_freedom": degrees_of_freedom,
                       "cycle_p_value": p_value,
                       "status": tension_status(p_value) if p_value is not None else "ok"})
    return scales, groups


def _gaussian_elimination(matrix, rhs):
    n = len(matrix)
    a = [row[:] for row in matrix]
    b = rhs[:]
    for column in range(n):
        pivot_row = max(range(column, n), key=lambda row: abs(a[row][column]))
        if abs(a[pivot_row][column]) < 1e-15:
            raise ValueError("Singular scale system")
        a[column], a[pivot_row] = a[pivot_row], a[column]
        b[column], b[pivot_row] = b[pivot_row], b[column]
        pivot = a[column][column]
        for row in range(column + 1, n):
            factor = a[row][column] / pivot
            a[row][column] = 0.0
            for k in range(column + 1, n):
                a[row][k] -= factor * a[column][k]
            b[row] -= factor * b[column]
    solution = [0.0] * n
    for row in range(n - 1, -1, -1):
        solution[row] = (b[row] - sum(a[row][k] * solution[k] for k in range(row + 1, n))) / a[row][row]
    return solution


def estimate_scales(batch_classes):
    """Fit pairwise scale ratios over the overlap graph and propagate scales."""
    batch_count = len(batch_classes)
    pair_entries = []
    for classes in batch_classes:
        entries = []
        for key, item in classes.items():
            if item["estimate"] is not None:
                entries.append((key, item["estimate"]["mean"], item["estimate"]["variance"]))
        pair_entries.append(entries)
    edge_records = []
    unsupported_edges = []
    scale_outliers = []
    for i in range(batch_count):
        for j in range(i + 1, batch_count):
            result = pair_scale(pair_entries[i], pair_entries[j])
            if not result["supported"]:
                unsupported_edges.append({
                    "batch_i": i, "batch_j": j,
                    "n_shared": result["n_classes"],
                    "reason": result["reason"],
                })
                continue
            for item in result["outliers"]:
                if item["outlier"]:
                    scale_outliers.append({
                        "batch_i": i, "batch_j": j,
                        "class": item["class"],
                        "standardized_residual": item["standardized_residual"],
                        "residual": item["residual"],
                        "residual_sigma": item["residual_sigma"],
                    })
            edge_records.append({
                "i": i, "j": j,
                "rho": result["rho"],
                "relative_standard_error": result["relative_standard_error"],
                "n_classes": result["n_classes"],
                "n_shared": result["n_shared"],
                "chi_squared": result["chi_squared"],
                "degrees_of_freedom": result["degrees_of_freedom"],
                "p_value": result["p_value"],
                "status": result["status"],
                "excluded_classes": [list(key) for key in sorted(result["robust_outlier_keys"])],
            })
    final_scales, final_groups = solve_scales(batch_count, edge_records)
    return {
        "scales": final_scales,
        "groups": final_groups,
        "edges": edge_records,
        "unsupported_edges": unsupported_edges,
        "scale_outliers": scale_outliers,
    }
