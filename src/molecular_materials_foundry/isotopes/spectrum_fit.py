"""Observed-spectrum fitting against known candidate isotope compositions.

Unlike ``instrument_envelope`` (which only emits a longer theoretical peak
list), this component fits a measured centroid spectrum through one shared
observation caliber: theoretical states are broadened into the same bins as
the observed peaks, window censoring and non-detections enter the likelihood
directly, and unresolved states contribute a quantified merge uncertainty.
Probability mass outside the window is never renormalized back in.
"""
import math
from ..validation import fields, mapping, sequence, label, integer, number
from ..errors import DomainError
from . import forward
from .fitting import (Budget, BudgetExhausted, golden_minimum, simplex_fit,
                      coordinate_fit, mixture_nll, inverse_normal_cdf, profile_interval)

MAX_CANDIDATES = 8
MAX_OBSERVATIONS = 200
OUTER_CYCLES = 3


def process(payload):
    request = _parse(payload)
    budget = Budget(request["budget_evaluations"], request["budget_seconds"])
    budget.begin()
    reasons = []
    problem = _Problem(request, budget)
    point = problem.initial_point()
    nll_min = math.inf
    try:
        # Multi-start: the uniform simplex point plus one corner per candidate,
        # because coordinate descent cannot revive a share collapsed to zero.
        starts = [point] + [problem.corner_point(j) for j in range(problem.n)]
        best_point, best_nll = None, math.inf
        for start in starts:
            candidate_point, candidate_nll = problem.full_fit(start)
            if candidate_nll < best_nll:
                best_point, best_nll = candidate_point, candidate_nll
        point, nll_min = problem.best
    except BudgetExhausted as exc:
        reasons.append("fit_" + exc.reason)
        if problem.best is None:
            nll_min = problem.nll(point[0], point[1], point[2], count=False)
            problem.best = (point, nll_min)
        point, nll_min = problem.best
    groups, rank = problem.design_diagnostics()
    intervals, interval_failed = problem.profile(point, nll_min, groups)
    if interval_failed:
        reasons.append("interval_" + interval_failed)
    return problem.assemble(point, nll_min, groups, rank, intervals, reasons, budget)


# --------------------------------------------------------------------------- #
# Request parsing
# --------------------------------------------------------------------------- #

def _parse(payload):
    fields(payload, ["candidates", "instrument", "observations"],
           ["charge", "adduct_mass", "window", "detection_limit", "background",
            "likelihood", "confidence", "budget", "grid_points"])
    charge = integer(payload.get("charge", 0), "charge", minimum=-20, maximum=20)
    adduct = number(payload.get("adduct_mass", 0.0), "adduct mass", minimum=-1000, maximum=100000)
    candidates = _parse_candidates(sequence(payload["candidates"], "candidates", minimum=1, maximum=MAX_CANDIDATES))
    instrument = _parse_instrument(mapping(payload["instrument"], "instrument"))
    observations, likelihood, detection, background, window = _parse_observations(payload, instrument)
    confidence = number(payload.get("confidence", 0.95), "confidence", minimum=0.5, maximum=0.999999)
    grid_points = integer(payload.get("grid_points", 17), "grid points", minimum=5, maximum=100)
    budget_obj = mapping(payload.get("budget", {}), "budget")
    fields(budget_obj, [], ["max_evaluations", "max_seconds"])
    max_evaluations = integer(budget_obj.get("max_evaluations", 60000), "budget evaluations", minimum=100, maximum=2000000)
    max_seconds = number(budget_obj.get("max_seconds", 10.0), "budget seconds", minimum=0.01, maximum=120.0)
    return {"candidates": candidates, "charge": charge, "adduct": adduct, "instrument": instrument,
            "observations": observations, "likelihood": likelihood, "detection": detection,
            "background": background, "window": window, "confidence": confidence,
            "grid_points": grid_points, "budget_evaluations": max_evaluations, "budget_seconds": max_seconds}


def _parse_candidates(raw):
    candidates = []
    ids = set()
    for item in raw:
        mapping(item, "candidate")
        fields(item, ["id", "composition"], ["populations", "labeling"])
        candidate_id = label(item["id"], "candidate id")
        if candidate_id in ids:
            raise DomainError("duplicate_candidate", "Candidate ids must be unique")
        ids.add(candidate_id)
        composition = mapping(item["composition"], "composition")
        populations = mapping(item.get("populations", {}), "populations")
        labeling = []
        for spec in sequence(item.get("labeling", []), "labeling", minimum=0, maximum=8):
            mapping(spec, "labeling rule")
            fields(spec, ["element", "labeled_isotope", "bounds"], [])
            element = label(spec["element"], "labeling element")
            isotope_index = integer(spec["labeled_isotope"], "labeled isotope index", minimum=0, maximum=7)
            bounds = sequence(spec["bounds"], "labeling bounds", minimum=2, maximum=2)
            low = number(bounds[0], "labeling lower bound", minimum=0.0, maximum=1.0)
            high = number(bounds[1], "labeling upper bound", minimum=0.0, maximum=1.0)
            if not low < high:
                raise DomainError("invalid_labeling_bounds", "Labeling bounds must be strictly ordered")
            if element not in composition:
                raise DomainError("labeling_element", "Labeling rules must reference a composition element")
            labeling.append({"element": element, "labeled_isotope": isotope_index,
                             "low": low, "high": high})
        if len({spec["element"] for spec in labeling}) != len(labeling):
            raise DomainError("duplicate_labeling", "At most one labeling rule per element")
        candidates.append({"id": candidate_id, "composition": composition,
                           "populations": populations, "labeling": labeling})
    return candidates


def _parse_instrument(raw):
    fields(raw, ["resolution"], ["resolution_at", "scaling", "rule", "merge_factor"])
    resolution = number(raw["resolution"], "resolution", positive=True, maximum=1e9)
    resolution_at = raw.get("resolution_at")
    if resolution_at is not None:
        resolution_at = number(resolution_at, "reference m/z", positive=True, maximum=1e7)
    scaling = raw.get("scaling", "constant")
    if scaling not in ("constant", "sqrt_mz"):
        raise DomainError("invalid_scaling", "scaling must be constant or sqrt_mz")
    if scaling == "sqrt_mz" and resolution_at is None:
        raise DomainError("missing_resolution_at", "sqrt_mz scaling requires a reference m/z")
    rule = raw.get("rule", "fwhm")
    if rule not in ("fwhm", "width"):
        raise DomainError("invalid_resolution_rule", "rule must be fwhm or width")
    merge_factor = number(raw.get("merge_factor", 1.0), "merge factor", positive=True, maximum=10.0)
    return forward.Instrument(resolution, resolution_at, scaling, rule, merge_factor)


def _parse_observations(payload, instrument):
    raw = sequence(payload["observations"], "observations", minimum=0, maximum=MAX_OBSERVATIONS)
    likelihood = payload.get("likelihood")
    if likelihood not in (None, "poisson", "gaussian"):
        raise DomainError("invalid_likelihood", "likelihood must be poisson or gaussian")
    has_error = any(isinstance(mapping(item, "observation").get("error"), (int, float)) for item in raw)
    if likelihood is None:
        likelihood = "gaussian" if has_error else "poisson"
    if likelihood == "gaussian" and not has_error:
        raise DomainError("missing_error", "Gaussian likelihood requires an error on each observation")
    window = payload.get("window")
    if window is not None:
        bounds = sequence(window, "window", minimum=2, maximum=2)
        low = number(bounds[0], "window lower edge", positive=True)
        high = number(bounds[1], "window upper edge", positive=True)
        if not low < high:
            raise DomainError("invalid_window", "Window edges must be strictly ordered")
        window = (low, high)
    detection = payload.get("detection_limit")
    if detection is not None:
        detection = number(detection, "detection limit", minimum=0.0)
    background = number(payload.get("background", 0.0), "background", minimum=0.0)
    observations = []
    seen = set()
    for item in raw:
        mapping(item, "observation")
        fields(item, ["coordinate", "intensity"], ["error"])
        coordinate = number(item["coordinate"], "observed m/z", positive=True, maximum=1e7)
        if window is not None and not window[0] <= coordinate <= window[1]:
            raise DomainError("observation_outside_window", "Every observed peak must lie inside the window")
        if coordinate in seen:
            raise DomainError("duplicate_observation", "Observed m/z values must be unique")
        seen.add(coordinate)
        intensity = item["intensity"]
        if likelihood == "poisson":
            intensity = integer(intensity, "observed count", minimum=0, maximum=10**12)
            if "error" in item:
                raise DomainError("unexpected_error", "Poisson likelihood derives uncertainty from counts")
        else:
            if "error" not in item:
                raise DomainError("missing_error", "Gaussian likelihood requires an error on each observation")
            intensity = number(intensity, "observed intensity", minimum=0.0)
            error = number(item["error"], "observation error", positive=True)
        observations.append({"coordinate": coordinate, "intensity": intensity,
                             "error": item.get("error") if likelihood == "gaussian" else None})
    return observations, likelihood, detection, background, window


# --------------------------------------------------------------------------- #
# Problem: design matrix, likelihood, optimization, profiles
# --------------------------------------------------------------------------- #

class _Problem:
    def __init__(self, request, budget):
        self.request = request
        self.budget = budget
        self.instrument = request["instrument"]
        self.candidates = request["candidates"]
        self.n = len(self.candidates)
        self.label_specs = [(j, k, spec) for j, candidate in enumerate(self.candidates)
                            for k, spec in enumerate(candidate["labeling"])]
        self._state_cache = {}
        initial_fractions = self._initial_fractions()
        per_candidate_states = [self._states(j, initial_fractions[j]) for j in range(self.n)]
        cluster_centroids = [cluster["centroid"] for states in per_candidate_states
                             for cluster in forward.merge_clusters(states, self.instrument)]
        observed_coordinates = [item["coordinate"] for item in request["observations"]]
        self.centers, self.lower, self.upper = forward.build_bins(
            observed_coordinates, cluster_centroids, request["window"], self.instrument)
        self.nbins = len(self.centers)
        if self.nbins == 0:
            raise DomainError("empty_observation_caliber",
                              "No observed peaks and no theoretical mass inside any window")
        self.observed_index = {}
        for coordinate in observed_coordinates:
            self.observed_index[coordinate] = self._bin_index(coordinate)
        self.observations = [0.0] * self.nbins
        self.variances = [1.0] * self.nbins
        self.censored = [False] * self.nbins
        self.detection = [0] * self.nbins
        for item in request["observations"]:
            b = self.observed_index[item["coordinate"]]
            self.observations[b] = float(item["intensity"])
            if request["likelihood"] == "gaussian":
                self.variances[b] = item["error"] ** 2
        limit = request["detection"]
        observed_bins = set(self.observed_index.values())
        self.observed_bins = observed_bins
        for b, center in enumerate(self.centers):
            if b not in observed_bins:
                if limit is None:
                    if request["likelihood"] == "gaussian":
                        raise DomainError("missing_detection_limit",
                                          "Theoretical bins without an observed peak require a detection_limit")
                    # Poisson default: an unreported bin is a zero count.
                    self.censored[b] = True
                    self.detection[b] = 0
                else:
                    self.censored[b] = True
                    self.detection[b] = math.floor(limit) if request["likelihood"] == "poisson" else limit
            elif limit is not None and self.observations[b] <= limit:
                self.censored[b] = True
                self.detection[b] = math.floor(limit) if request["likelihood"] == "poisson" else limit
        self.merge_info = forward.bin_merge_uncertainty(
            per_candidate_states, self.centers, self.lower, self.upper, self.instrument)
        self.best = None
        self.columns_at(initial_fractions)  # prime the cache and diagnostics

    def _bin_index(self, coordinate):
        for b, center in enumerate(self.centers):
            if center == coordinate or abs(center - coordinate) <= 1e-10 * abs(coordinate):
                return b
        raise DomainError("observation_outside_window", "Observed peak could not be assigned to a bin")

    # -- candidate theory --------------------------------------------------- #

    def _initial_fractions(self):
        return [{spec["element"]: 0.5 * (spec["low"] + spec["high"]) for spec in c["labeling"]}
                for c in self.candidates]

    def _states(self, j, fractions):
        candidate = self.candidates[j]
        key = (j, tuple(sorted((element, round(value, 8)) for element, value in fractions.items())))
        if key not in self._state_cache:
            labeled = [dict(spec, fraction=fractions[spec["element"]]) for spec in candidate["labeling"]]
            payload = dict(candidate, labeling=labeled)
            self._state_cache[key] = forward.candidate_states(
                payload, self.request["charge"], self.request["adduct"])
        return self._state_cache[key]

    def columns_at(self, fractions):
        columns, outside, cross = [], [], []
        for j in range(self.n):
            states = self._states(j, fractions[j])
            column, tail, spill = forward.design_column(
                states, self.centers, self.lower, self.upper, self.request["window"], self.instrument)
            columns.append(column)
            outside.append(tail)
            cross.append(spill)
        self._columns = columns
        self._outside = outside
        self._cross = cross
        self._fractions = fractions
        return columns

    def window_mass(self, j, fractions):
        window = self.request["window"]
        states = self._states(j, fractions[j])
        if window is None:
            return 1.0, 0.0
        inside = math.fsum(probability for mass, probability in states if window[0] <= mass <= window[1])
        return inside, 1.0 - inside

    # -- likelihood --------------------------------------------------------- #

    def _means(self, shares, scale, columns):
        background = self.request["background"]
        return [background + scale * math.fsum(shares[j] * columns[j][b] for j in range(self.n))
                for b in range(self.nbins)]

    def nll(self, fractions, shares, scale, count=True):
        if count:
            self.budget.tick()
        columns = self.columns_at(fractions)
        means = self._means(shares, scale, columns)
        return mixture_nll(means, self.observations, self.variances, self.censored,
                           self.detection, self.request["likelihood"])

    def initial_point(self):
        fractions = self._initial_fractions()
        shares = [1.0 / self.n] * self.n
        scale = max(1.0, math.fsum(self.observations))
        return fractions, shares, scale

    def corner_point(self, dominant):
        fractions = self._initial_fractions()
        shares = [0.0] * self.n
        shares[dominant] = 1.0
        scale = max(1.0, math.fsum(self.observations))
        return fractions, shares, scale

    def _scale_bounds(self):
        total = math.fsum(self.observations)
        # The upper edge must stay finite for bounded optimization, but has to
        # accommodate spectra where most probability is censored outside the window.
        return 1e-9, max(1.0, total * 1000.0)

    def full_fit(self, point):
        fractions, shares, scale = point
        for _ in range(OUTER_CYCLES):
            previous = (shares, scale)
            shares = self._fit_shares(fractions, shares, scale, fixed={})
            scale = self._fit_scale(fractions, shares, scale)
            for j, k, spec in self.label_specs:
                fractions, shares, scale = self._fit_label(fractions, shares, scale, j, spec)
            shares = self._fit_shares(fractions, shares, scale, fixed={})
            scale = self._fit_scale(fractions, shares, scale)
            if all(abs(a - b) < 1e-6 for a, b in zip(shares, previous[0])) and abs(scale - previous[1]) < 1e-6:
                break
        nll = self.nll(fractions, shares, scale)
        self._remember(fractions, shares, scale, nll)
        return (fractions, shares, scale), nll

    def _fit_shares(self, fractions, shares, scale, fixed, max_cycles=10):
        def objective(trial_shares):
            value = self.nll(fractions, trial_shares, scale)
            self._remember(fractions, trial_shares, scale, value)
            return value
        return simplex_fit(objective, self.n, fixed, shares, max_cycles=max_cycles)

    def _fit_scale(self, fractions, shares, scale, max_cycles=3):
        def objective(value):
            result = self.nll(fractions, shares, value[0])
            self._remember(fractions, shares, value[0], result)
            return result
        low, high = self._scale_bounds()
        return coordinate_fit(objective, [scale], [(low, high)], 1e-6, max_cycles=max_cycles)[0]

    def _fit_label(self, fractions, shares, scale, j, spec):
        fractions = [dict(mapping) for mapping in fractions]

        def objective(value):
            trial = [dict(mapping) for mapping in fractions]
            trial[j][spec["element"]] = value
            trial_shares = self._fit_shares(trial, shares, scale, {}, max_cycles=4)
            trial_scale = self._fit_scale(trial, trial_shares, scale, max_cycles=2)
            result = self.nll(trial, trial_shares, trial_scale)
            self._remember(trial, trial_shares, trial_scale, result)
            return result

        value = golden_minimum(objective, spec["low"], spec["high"], 1e-5)[0]
        fractions[j][spec["element"]] = value
        shares = self._fit_shares(fractions, shares, scale, {})
        scale = self._fit_scale(fractions, shares, scale)
        return fractions, shares, scale

    def _remember(self, fractions, shares, scale, value):
        if self.best is None or value < self.best[1]:
            self.best = (([dict(m) for m in fractions], list(shares), scale), value)

    # -- diagnostics -------------------------------------------------------- #

    def design_diagnostics(self):
        columns = self._columns
        groups = self._indistinguishable(columns)
        rank = self._rank(columns)
        return groups, rank

    def _indistinguishable(self, columns):
        normalized = []
        for column in columns:
            mass = math.fsum(column)
            normalized.append([v / mass for v in column] if mass > 0 else column)
        unions = list(range(self.n))

        def find(x):
            while unions[x] != x:
                unions[x] = unions[unions[x]]
                x = unions[x]
            return x

        def union(a, b):
            unions[find(a)] = find(b)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                distance = math.fsum(abs(a - b) for a, b in zip(normalized[i], normalized[j]))
                if distance < 1e-3:
                    union(i, j)
        grouped = {}
        for i in range(self.n):
            grouped.setdefault(find(i), []).append(i)
        return [members for members in grouped.values() if len(members) > 1]

    def _rank(self, columns, tolerance=1e-9):
        basis = []
        for column in columns:
            vector = list(column)
            for row in basis:
                projection = math.fsum(v * r for v, r in zip(vector, row))
                vector = [v - projection * r for v, r in zip(vector, row)]
            norm = math.sqrt(math.fsum(v * v for v in vector))
            if norm > tolerance:
                basis.append([v / norm for v in vector])
        return len(basis)

    # -- profile intervals -------------------------------------------------- #

    def profile(self, point, nll_min, groups):
        fractions, shares, scale = point
        z = inverse_normal_cdf(0.5 + 0.5 * self.request["confidence"])
        delta = 0.5 * z * z
        grid = self.request["grid_points"]
        intervals = {"shares": {}, "labels": {}, "scale": None}
        failed = None
        grouped_members = {i for members in groups for i in members}
        try:
            # Share profiles, collapsing indistinguishable groups.
            targets = self._group_targets(groups, grouped_members, shares)
            for key, fixed_builder, collapsed_n, start_shares, low_high in targets:
                def nll_at(value, builder=fixed_builder):
                    fixed = builder(value)
                    seed = list(start_shares)
                    for index, fixed_value in fixed.items():
                        seed[index] = fixed_value
                    trial_shares = simplex_fit(
                        lambda trial: self.nll(fractions, trial, scale),
                        collapsed_n, set(fixed), seed, max_cycles=6)
                    return self.nll(fractions, trial_shares, scale)
                minimum = math.fsum(shares[i] for i in self._id_index(key))
                intervals["shares"][key] = profile_interval(
                    nll_at, minimum, low_high[0], low_high[1], nll_min, delta, grid)
            for j, k, spec in self.label_specs:
                def nll_at(value, candidate=j, rule=spec):
                    trial = [dict(m) for m in fractions]
                    trial[candidate][rule["element"]] = value
                    trial_shares = self._fit_shares(trial, shares, scale, {}, max_cycles=6)
                    trial_scale = self._fit_scale(trial, trial_shares, scale, max_cycles=2)
                    return self.nll(trial, trial_shares, trial_scale)
                intervals["labels"][(j, k)] = profile_interval(
                    nll_at, fractions[j][spec["element"]], spec["low"], spec["high"],
                    nll_min, delta, grid)
            intervals["scale"] = self._profile_scale(fractions, shares, scale, nll_min, delta, grid)
        except BudgetExhausted as exc:
            failed = exc.reason
        return intervals, failed

    def _group_targets(self, groups, grouped_members, shares):
        targets = []
        for members in groups:
            key = tuple(members)

            def builder(value, members=members):
                # Equal split is unidentifiable; the profile is on the group total.
                return {i: value / len(members) for i in members}
            targets.append((key, builder, self.n, list(shares), (0.0, 1.0)))
        for j in range(self.n):
            if j not in grouped_members:
                targets.append((j,
                                lambda value, index=j: {index: value},
                                self.n, list(shares), (0.0, 1.0)))
        return targets

    def _id_index(self, key):
        if isinstance(key, int):
            return [key]
        return list(key)

    def _profile_scale(self, fractions, shares, scale, nll_min, delta, grid):
        low_bound, high_bound = self._scale_bounds()
        # Profile on an adaptive span around the MLE so grid resolution is not
        # wasted across the full (possibly very wide) numeric scale bound.
        low = max(low_bound, scale * 0.25)
        high = min(high_bound, max(scale * 4.0, scale + 1.0))

        def nll_at(value):
            return self.nll(fractions, shares, value)
        return profile_interval(nll_at, scale, low, high, nll_min, delta, grid)

    # -- response ----------------------------------------------------------- #

    def assemble(self, point, nll_min, groups, rank, intervals, reasons, budget):
        fractions, shares, scale = point
        columns = self.columns_at(fractions)
        means = self._means(shares, scale, columns)
        likelihood = self.request["likelihood"]
        residuals = []
        chi_squared = 0.0
        unmatched = []
        non_detections = []
        for b, center in enumerate(self.centers):
            if likelihood == "gaussian":
                sigma = math.sqrt(self.variances[b])
                pearson = (self.observations[b] - means[b]) / sigma
            else:
                sigma = math.sqrt(max(means[b], 1e-300))
                pearson = (self.observations[b] - means[b]) / sigma
            if not self.censored[b]:
                chi_squared += pearson ** 2
            record = {
                "coordinate": center,
                "observed": None if self.censored[b] else self.observations[b],
                "fitted": means[b],
                "residual": None if self.censored[b] else self.observations[b] - means[b],
                "pearson_residual": pearson,
                "censored": self.censored[b],
                "detection_limit": self.detection[b] if self.censored[b] else None,
                "merge_sigma_mz": self.merge_info[b]["sigma_mz"],
                "merged_state_count": self.merge_info[b]["state_count"],
            }
            residuals.append(record)
            if self.censored[b]:
                non_detections.append(center)
            if b in self.observed_bins and not self._has_nearby_theory(b, columns):
                unmatched.append(center)
        candidate_records = []
        grouped_members = {i for members in groups for i in members}
        for j, candidate in enumerate(self.candidates):
            inside, outside_window = self.window_mass(j, fractions)
            labels = []
            for k, spec in enumerate(candidate["labeling"]):
                interval = intervals["labels"].get((j, k))
                labels.append({
                    "element": spec["element"],
                    "labeled_isotope": spec["labeled_isotope"],
                    "bounds": [spec["low"], spec["high"]],
                    "fraction": fractions[j][spec["element"]],
                    "fraction_interval": _interval_or_none(interval),
                })
            share_interval = None
            if j not in grouped_members:
                share_interval = _interval_or_none(intervals["shares"].get(j))
            candidate_records.append({
                "id": candidate["id"],
                "share": shares[j],
                "share_interval": share_interval,
                "indistinguishable_group": next(
                    (g for g, members in enumerate(groups) if j in members), None),
                "labeling": labels,
                "state_count": len(self._states(j, fractions[j])),
                "probability_inside_window": inside,
                "outside_window_probability": outside_window,
                "tail_outside_bins": self._outside[j],
                "cross_bin_spill": self._cross[j],
            })
        group_records = []
        for g, members in enumerate(groups):
            group_records.append({
                "candidate_ids": [self.candidates[i]["id"] for i in members],
                "total_share": math.fsum(shares[i] for i in members),
                "share_interval": _interval_or_none(intervals["shares"].get(tuple(members))),
                "note": ("Candidates are not separately identifiable; only the group total is fitted. "
                         "The per-candidate share values are an arbitrary equal split of the group total"),
            })
        free_parameters = rank + 1 + len(self.label_specs)
        observed_bins = self.nbins - len(non_detections)
        truncated = [key for key, interval in intervals["shares"].items()
                     if interval and (interval["truncated_low"] or interval["truncated_high"])]
        truncated += [(j, k) for (j, k), interval in intervals["labels"].items()
                      if interval and (interval["truncated_low"] or interval["truncated_high"])]
        # The scale lower edge is a numeric floor, so its truncation is not reported
        # as an identification limit; only upper-edge truncation matters there.
        if intervals["scale"] and intervals["scale"]["truncated_high"]:
            truncated.append("scale")
        if reasons:
            status = "incomplete"
        elif rank < self.n or truncated:
            status = "completed_with_limits"
            if rank < self.n:
                reasons.append("design_rank_below_candidate_count")
            if truncated:
                reasons.append("interval_touched_parameter_bound")
        else:
            status = "completed"
        return {
            "model": "isotope-spectrum-fit-v1",
            "status": status,
            "status_reasons": reasons,
            "coordinate": "mass_to_charge" if self.request["charge"] else "neutral_mass",
            "charge": self.request["charge"],
            "adduct_mass": self.request["adduct"],
            "likelihood": likelihood,
            "confidence": self.request["confidence"],
            "window": list(self.request["window"]) if self.request["window"] else None,
            "candidates": candidate_records,
            "indistinguishable_groups": group_records,
            "scale": {
                "value": scale,
                "interval": _interval_or_none(intervals["scale"]),
                "interpretation": "expected total ion counts" if likelihood == "poisson"
                else "intensity per unit mixture probability",
            },
            "residuals": residuals,
            "non_detection_bins": non_detections,
            "unmatched_observed_bins": unmatched,
            "fit": {
                "negative_log_likelihood": nll_min,
                "chi_squared": chi_squared,
                "observed_bins": observed_bins,
                "censored_bins": len(non_detections),
                "degrees_of_freedom": max(0, observed_bins - free_parameters),
                "free_parameters": free_parameters,
            },
            "design": {
                "candidate_count": self.n,
                "rank": rank,
                "bin_count": self.nbins,
            },
            "error_budget": {
                "window_censored_probability": [rec["outside_window_probability"] for rec in candidate_records],
                "tail_outside_bins": [rec["tail_outside_bins"] for rec in candidate_records],
                "cross_bin_spill": [rec["cross_bin_spill"] for rec in candidate_records],
                "merge_sigma_mz_max": max(item["sigma_mz"] for item in self.merge_info),
                "merge_sigma_mz_mean": math.fsum(item["sigma_mz"] for item in self.merge_info) / self.nbins,
                "enumeration_truncated_probability": 0.0,
                "renormalized": False,
            },
            "budget": {
                "evaluations": budget.evaluations,
                "max_evaluations": budget.max_evaluations,
                "elapsed_seconds": budget.elapsed(),
                "max_seconds": budget.max_seconds,
            },
        }

    def _has_nearby_theory(self, b, columns):
        center = self.centers[b]
        for j in range(self.n):
            if columns[j][b] > 1e-12:
                return True
            for mass, probability in self._states(j, self._fractions[j]):
                if probability > 0 and abs(mass - center) < self.instrument.width(center):
                    return True
        return False


def _interval_or_none(interval):
    if interval is None:
        return None
    return {key: value for key, value in interval.items()}
