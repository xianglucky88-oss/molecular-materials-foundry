"""Forward observation model for isotope-spectrum fitting.

The exact isotope enumeration from :mod:`mass_distribution` is reused unchanged.
This module adds the layers that turn theoretical states into the same
observation caliber as a measured centroid spectrum:

* ion conversion with a global charge and one adduct mass,
* variable isotope-labeling populations,
* instrument resolution widths, unresolved-peak merging and Gaussian tails,
* observation bins, window censoring and probability that spills outside.

No renormalization is performed after censoring: a candidate's design column
sums to its probability mass inside the window, never to one.
"""
import math
from bisect import bisect_left, bisect_right
from .populations import prepare, ELECTRON_MASS
from .mass_distribution import enumerate_distribution
from ..errors import DomainError

FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
TAIL_SIGMA = 6.0


def normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def ion_coordinate(neutral_mass, charge, adduct_mass):
    ion = neutral_mass + adduct_mass - charge * ELECTRON_MASS
    if ion <= 0:
        raise DomainError("invalid_ion_mass", "The declared charge or adduct produces a nonpositive ion mass")
    return ion / abs(charge) if charge else ion


def apply_labeling(groups, specs):
    """Replace one isotope's probability with a variable labeled fraction."""
    by_symbol = {symbol: (count, values) for symbol, count, values in groups}
    seen = set()
    for spec in specs:
        symbol = spec["element"]
        if symbol in seen:
            raise DomainError("duplicate_labeling", "At most one labeling rule per element")
        seen.add(symbol)
        if symbol not in by_symbol:
            raise DomainError("labeling_element", "Labeling rules must reference a composition element")
        count, values = by_symbol[symbol]
        index = spec["labeled_isotope"]
        if not 0 <= index < len(values):
            raise DomainError("labeled_isotope_index", "Labeled isotope index is outside the population")
        base = [probability for _, probability in values]
        if base[index] >= 1.0:
            raise DomainError("labeling_deterministic", "A deterministic isotope population has no unlabeled fraction to vary")
        fraction = spec["fraction"]
        remainder = 1.0 - base[index]
        updated = []
        for i, probability in enumerate(base):
            if i == index:
                updated.append(fraction)
            else:
                updated.append((1.0 - fraction) * probability / remainder)
        shift = 1.0 - math.fsum(updated)
        updated[index] += shift
        by_symbol[symbol] = (count, tuple((mass, prob) for (mass, _), prob in zip(values, updated)))
    return [(symbol, count, values) for symbol, (count, values) in sorted(by_symbol.items())]


def candidate_states(candidate, charge, adduct_mass):
    groups, _ = prepare({
        "composition": candidate["composition"],
        "populations": candidate.get("populations", {}),
        "charge": 0,
    })
    groups = apply_labeling(groups, candidate.get("labeling", ()))
    states = enumerate_distribution(groups)
    return [(ion_coordinate(mass, charge, adduct_mass), probability) for mass, probability in states]


class Instrument:
    """Resolution rule giving a merge width and Gaussian sigma at any m/z."""

    def __init__(self, resolution, resolution_at, scaling, rule, merge_factor):
        self.resolution = resolution
        self.resolution_at = resolution_at
        self.scaling = scaling
        self.rule = rule
        self.merge_factor = merge_factor

    def resolution_at_mz(self, mz):
        if self.resolution_at is None or self.scaling == "constant":
            return self.resolution
        return self.resolution * math.sqrt(mz / self.resolution_at)

    def width(self, mz):
        return self.merge_factor * mz / self.resolution_at_mz(mz)

    def sigma(self, mz):
        width = self.width(mz)
        return width / FWHM_FACTOR if self.rule == "fwhm" else width


def merge_clusters(states, instrument):
    """Merge states too close to resolve, keeping the centroid spread."""
    ordered = sorted(states)
    clusters = []
    current = []
    for state in ordered:
        if current and state[0] - current[-1][0] < instrument.width(0.5 * (state[0] + current[-1][0])):
            current.append(state)
        else:
            if current:
                clusters.append(_summarize_cluster(current))
            current = [state]
    if current:
        clusters.append(_summarize_cluster(current))
    return clusters


def _summarize_cluster(current):
    total = math.fsum(probability for _, probability in current)
    centroid = math.fsum(mass * probability for mass, probability in current) / total
    spread = math.sqrt(max(0.0, math.fsum(probability * (mass - centroid) ** 2 for mass, probability in current) / total))
    return {"centroid": centroid, "probability": total, "spread": spread, "state_count": len(current)}


def build_bins(observed_coordinates, cluster_centroids, window, instrument):
    raw = sorted(set(observed_coordinates) | {
        center for center in cluster_centroids
        if window is None or window[0] <= center <= window[1]
    })
    # Snap together centers that differ only by floating-point jitter (relative
    # 1e-10), so an observed peak and its theoretical target share one bin.
    observed_set = set(observed_coordinates)
    centers = []
    for center in raw:
        if centers and centers[-1] != center and abs(center - centers[-1]) <= 1e-10 * abs(center):
            if centers[-1] not in observed_set and center in observed_set:
                centers[-1] = center
            continue
        centers.append(center)
    if not centers:
        return [], [], []
    if window is None:
        # Without an explicit window the outermost bins reach out to the Gaussian
        # tail cutoff, so the design column conserves essentially all probability.
        left_edge = min(centers[0] - TAIL_SIGMA * instrument.sigma(centers[0]), 0.0)
        right_edge = centers[-1] + TAIL_SIGMA * instrument.sigma(centers[-1])
    else:
        left_edge, right_edge = window
    boundaries = []
    for i, center in enumerate(centers):
        if i == 0:
            left = left_edge
        else:
            left = 0.5 * (center + centers[i - 1])
        if i == len(centers) - 1:
            right = right_edge
        else:
            right = 0.5 * (center + centers[i + 1])
        boundaries.append((left, right))
    return centers, [left for left, _ in boundaries], [right for _, right in boundaries]


def design_column(states, centers, lower, upper, window, instrument):
    """Distribute one candidate's states over bins via Gaussian CDF differences."""
    nbins = len(centers)
    column = [0.0] * nbins
    outside = 0.0
    cross_spill = 0.0
    for mass, probability in states:
        sigma = instrument.sigma(mass)
        reach = TAIL_SIGMA * sigma
        first = max(0, bisect_left(upper, mass - reach) - 1)
        last = min(nbins, bisect_right(lower, mass + reach) + 1)
        inside = 0.0
        home = None
        home_mass = 0.0
        for b in range(first, last):
            fraction = normal_cdf((upper[b] - mass) / sigma) - normal_cdf((lower[b] - mass) / sigma)
            if fraction <= 0.0:
                continue
            contribution = probability * fraction
            column[b] += contribution
            inside += contribution
            if abs(centers[b] - mass) < home_mass or home is None:
                home = b
                home_mass = abs(centers[b] - mass)
        inside = min(probability, max(0.0, inside))
        outside += probability - inside
        if home is not None:
            own = min(probability, max(0.0, probability * (
                normal_cdf((upper[home] - mass) / sigma) - normal_cdf((lower[home] - mass) / sigma))))
            cross_spill += probability - own
    return column, outside, cross_spill


def bin_merge_uncertainty(per_candidate_states, centers, lower, upper, instrument):
    """Combined m/z uncertainty per bin: instrument width plus within-bin spread."""
    nbins = len(centers)
    weighted_mean = [0.0] * nbins
    weighted_weight = [0.0] * nbins
    weighted_square = [0.0] * nbins
    state_counts = [0] * nbins
    for states in per_candidate_states:
        for mass, probability in states:
            sigma = instrument.sigma(mass)
            reach = TAIL_SIGMA * sigma
            first = max(0, bisect_left(upper, mass - reach) - 1)
            last = min(nbins, bisect_right(lower, mass + reach) + 1)
            for b in range(first, last):
                fraction = normal_cdf((upper[b] - mass) / sigma) - normal_cdf((lower[b] - mass) / sigma)
                weight = probability * fraction
                if weight < 1e-12:
                    continue
                weighted_mean[b] += weight * mass
                weighted_square[b] += weight * mass * mass
                weighted_weight[b] += weight
                state_counts[b] += 1
    result = []
    for b, center in enumerate(centers):
        weight = weighted_weight[b]
        spread = 0.0
        if weight > 0.0:
            mean = weighted_mean[b] / weight
            spread = math.sqrt(max(0.0, weighted_square[b] / weight - mean * mean))
        combined = math.hypot(instrument.sigma(center), spread)
        result.append({"sigma_mz": combined, "state_count": state_counts[b], "spread_mz": spread})
    return result
