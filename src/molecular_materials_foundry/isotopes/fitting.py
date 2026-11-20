"""Bounded likelihood fitting for candidate isotope mixtures.

Pure standard-library optimizer (golden-section bounded line searches driven by
cyclic coordinate descent) plus profile-likelihood intervals. The likelihood
closure supplied by the caller is responsible for counting each evaluation
against the declared :class:`Budget`; exhaustion propagates as
:class:`BudgetExhausted` so the caller reports an incomplete fit rather than
presenting an unreliable answer.
"""
import math
from ..errors import DomainError
from .forward import normal_cdf

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


class BudgetExhausted(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Budget:
    def __init__(self, max_evaluations, max_seconds):
        self.max_evaluations = max_evaluations
        self.max_seconds = max_seconds
        self.evaluations = 0
        self.start = None

    def begin(self):
        import time
        self.start = time.monotonic()

    def tick(self):
        import time
        self.evaluations += 1
        if self.evaluations > self.max_evaluations:
            raise BudgetExhausted("evaluation_limit")
        if self.start is not None and time.monotonic() - self.start > self.max_seconds:
            raise BudgetExhausted("time_limit")

    def elapsed(self):
        import time
        return time.monotonic() - self.start


def golden_minimum(function, low, high, tolerance, max_iterations=24):
    """Bounded golden-section minimum; a coarse scan first guards multimodality."""
    if high <= low:
        return low, function(low)
    steps = 9
    best_x, best_y = low, function(low)
    for i in range(1, steps):
        x = low + (high - low) * i / steps
        y = function(x)
        if y < best_y:
            best_x, best_y = x, y
    a = max(low, best_x - (high - low) / steps)
    b = min(high, best_x + (high - low) / steps)
    c = b - GOLDEN * (b - a)
    d = a + GOLDEN * (b - a)
    fc, fd = function(c), function(d)
    iterations = 0
    while (iterations < max_iterations
           and abs(b - a) > tolerance * max(1.0, abs(a) + abs(b))):
        iterations += 1
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = function(d)
    middle = 0.5 * (a + b)
    candidates = ((best_x, best_y), (middle, function(middle)), (c, fc), (d, fd),
                  (a, function(a)), (b, function(b)))
    return min(candidates, key=lambda item: item[1])


def coordinate_fit(function, point, bounds, tolerance, max_cycles=12):
    """Cyclic coordinate descent over independent bounded parameters."""
    point = list(point)
    for _ in range(max_cycles):
        previous = list(point)
        for i, (low, high) in enumerate(bounds):
            if high <= low:
                point[i] = low
                continue

            def line(value, index=i):
                trial = list(point)
                trial[index] = value
                return function(trial)

            point[i] = golden_minimum(line, low, high, tolerance)[0]
        if all(abs(point[i] - previous[i]) <= tolerance * max(1.0, abs(point[i]))
               for i in range(len(point))):
            break
    return point


def simplex_line(point, index, fixed, free, value):
    """Set one simplex coordinate, rescaling the other free coordinates."""
    fixed_mass = math.fsum(point[i] for i in fixed)
    other_free = [i for i in free if i != index]
    other_sum = math.fsum(point[i] for i in other_free)
    trial = list(point)
    trial[index] = value
    room = max(0.0, 1.0 - fixed_mass - value)
    if other_sum <= 0.0:
        for i in other_free:
            trial[i] = room / len(other_free) if other_free else 0.0
    else:
        for i in other_free:
            trial[i] = point[i] * room / other_sum
    return trial


def simplex_fit(nll, n_amplitudes, fixed, start, tolerance=1e-5, max_cycles=12):
    """Fit nonnegative simplex coordinates with optional fixed coordinates."""
    fixed_indexes = sorted(fixed)
    free = [i for i in range(n_amplitudes) if i not in fixed]
    point = list(start)
    for _ in range(max_cycles):
        previous = list(point)
        fixed_mass = math.fsum(point[i] for i in fixed_indexes)
        for i in free:
            other_sum = math.fsum(point[j] for j in free if j != i)
            limit = max(0.0, 1.0 - fixed_mass)

            def line(value, index=i):
                return nll(simplex_line(point, index, fixed_indexes, free, value))

            point = simplex_line(point, i, fixed_indexes, free,
                                 golden_minimum(line, 0.0, limit, tolerance)[0])
        if all(abs(point[i] - previous[i]) <= tolerance for i in range(n_amplitudes)):
            break
    return point


def poisson_cdf_log(mean, threshold):
    """log P(Poisson(mean) <= threshold) for an integer detection threshold."""
    if threshold <= 0:
        return -mean
    term = math.exp(-min(mean, 700.0))
    total = term
    for k in range(1, threshold + 1):
        term *= mean / k
        total += term
    return math.log(max(total, 1e-300))


def mixture_nll(means, observations, variances, censored, detection, likelihood):
    total = 0.0
    if likelihood == "poisson":
        for b, mean in enumerate(means):
            if censored[b]:
                total -= poisson_cdf_log(mean, detection[b])
            else:
                total -= (observations[b] * math.log(max(mean, 1e-300))
                          - mean - math.lgamma(observations[b] + 1.0))
        return total
    for b, mean in enumerate(means):
        if censored[b]:
            total -= math.log(max(normal_cdf((detection[b] - mean) / math.sqrt(variances[b])), 1e-300))
        else:
            total += 0.5 * ((observations[b] - mean) ** 2 / variances[b]
                            + math.log(2.0 * math.pi * variances[b]))
    return total


def inverse_normal_cdf(probability):
    """Inverse standard-normal CDF by Newton iteration on erf."""
    if not 0.0 < probability < 1.0:
        raise DomainError("invalid_confidence", "Confidence must lie strictly between zero and one")
    target = 2.0 * probability - 1.0
    x = 0.0
    for _ in range(50):
        step = (math.erf(x) - target) / max((2.0 / math.sqrt(math.pi)) * math.exp(-x * x), 1e-300)
        x -= step
        if abs(step) < 1e-12:
            break
    return x * math.sqrt(2.0)


def profile_interval(nll_at, minimum, low, high, nll_min, delta, grid_points):
    """1-D profile likelihood interval from an equal-spaced grid with interpolation."""
    span = high - low
    xs = [low + span * i / grid_points for i in range(grid_points + 1)]
    ys = []
    for x in xs:
        if abs(x - minimum) < span / grid_points:
            ys.append(nll_min)
        else:
            ys.append(nll_at(x))
    threshold = nll_min + delta
    low_edge = high_edge = None
    for i in range(grid_points):
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        if y0 == y1 or not (min(y0, y1) <= threshold <= max(y0, y1)):
            continue
        crossing = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
        if crossing < minimum and low_edge is None:
            low_edge = crossing
        elif crossing > minimum and high_edge is None:
            high_edge = crossing
    return {
        "low": low if low_edge is None else low_edge,
        "high": high if high_edge is None else high_edge,
        "truncated_low": low_edge is None,
        "truncated_high": high_edge is None,
        "threshold_delta": delta,
        "grid_points": grid_points + 1,
    }
