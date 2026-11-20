"""Calibration checks against synthetic acquisitions with known truth.

Analysts can verify the estimator on a measured or simulated control sample
where the active coherence channels, their complex amplitudes and the noise
level are known.  Noise is circular complex Gaussian; the random generator is
seeded so checks are reproducible.  Time-domain synthesis and estimation both
run in bounded blocks so long reference traces do not dominate memory.

``noise_sigma`` is the per-real-component standard deviation: each real and
imaginary component is N(0, sigma^2), so the circular complex variance
E|e|^2 = 2 sigma^2, matching the acquisition ``variance`` field convention.
"""
import math
import random

from .estimation import estimate

DEFAULT_TRIALS = 200
DEFAULT_BLOCK_SIZE = 128
# Chi-square with 2 degrees of freedom, 95th percentile: -2 * ln(0.05).
_QUANTILE_95 = -2.0 * math.log(0.05)


def _noiseless(rows, amplitudes, candidates, width):
    trace = []
    for column in range(width):
        total = 0.0j
        for row in rows:
            value = 0.0j
            for channel, amplitude in zip(candidates, amplitudes):
                value += amplitude[column] * complex(
                    math.cos(math.radians(channel * row["phase"])),
                    math.sin(math.radians(channel * row["phase"])))
            total += value * complex(math.cos(math.radians(-row["receiver"])),
                                     math.sin(math.radians(-row["receiver"])))
        trace.append(total)
    return trace


def calibration_check(rows, candidates, amplitudes, *, noise_sigma, trials=DEFAULT_TRIALS,
                      seed=72341, block_size=DEFAULT_BLOCK_SIZE, target_channel=None):
    """Run Monte Carlo trials and summarize bias, RMSE and 95% coverage.

    ``rows`` supply phases, receiver phases and one seed ``mean`` trace whose
    length defines the time-domain width.  ``amplitudes`` gives the true
    complex amplitude of every candidate channel at every time sample.
    ``noise_sigma`` is the per-component standard deviation (each of the real
    and imaginary parts is N(0, sigma^2)); the supplied variance field is
    E|e|^2 = 2 sigma^2.
    """
    if target_channel is None:
        target_channel = candidates[0]
    if target_channel not in candidates:
        raise ValueError("target_channel must be one of the candidates")
    width = len(rows[0]["mean"])
    if any(len(amplitude) != width for amplitude in amplitudes):
        raise ValueError("Every true amplitude must span the full time domain")
    if not math.isfinite(noise_sigma) or noise_sigma <= 0:
        raise ValueError("noise_sigma must be positive")
    rng = random.Random(seed)
    basis = [[complex(math.cos(math.radians(q * row["phase"])),
                      math.sin(math.radians(q * row["phase"]))) for q in candidates] for row in rows]

    complex_variance = 2.0 * noise_sigma * noise_sigma
    clean_rows = [{"phase": row["phase"], "receiver": row["receiver"],
                   "mean": [None] * width, "variance": [complex_variance]} for row in rows]
    target_index = candidates.index(target_channel)
    sums = [0.0j for _ in range(width)]
    square_bias = [0.0 for _ in range(width)]
    coverage = [0 for _ in range(width)]
    singular = 0
    for _ in range(trials):
        trial_rows = []
        for row, clean in zip(rows, clean_rows):
            observed = []
            for column in range(width):
                signal = complex(rng.gauss(0.0, noise_sigma),
                                 rng.gauss(0.0, noise_sigma))
                for channel, amplitude in zip(candidates, amplitudes):
                    signal += amplitude[column] * complex(
                        math.cos(math.radians(channel * row["phase"])),
                        math.sin(math.radians(channel * row["phase"])))
                observed.append(signal * complex(math.cos(math.radians(-row["receiver"])),
                                                  math.sin(math.radians(-row["receiver"]))))
            trial_rows.append({**clean, "mean": observed})
        try:
            result = estimate(basis, trial_rows, candidates, block_size=block_size)
        except ValueError:
            singular += 1
            continue
        if target_index not in result["independent"]:
            singular += 1
            continue
        record = result["records"][target_index]
        for column, estimate_value in enumerate(record["estimate"]):
            truth = amplitudes[target_index][column]
            sums[column] += estimate_value
            difference = estimate_value - truth
            square_bias[column] += abs(difference) ** 2
            gram_diagonal = record["gram_diagonal"][column]
            # Component noise sd is sigma with complex variance v = 2 sigma^2.
            # The Gram diagonal equals twice a component estimate variance,
            # so |delta|^2 / (G/2) is chi-square_2.
            complex_estimate_variance = 0.5 * gram_diagonal
            metric = (difference.real ** 2 + difference.imag ** 2) / complex_estimate_variance \
                if complex_estimate_variance > 0 else math.inf
            if metric <= _QUANTILE_95:
                coverage[column] += 1

    effective_trials = trials - singular
    if effective_trials:
        mean_estimate = [value / effective_trials for value in sums]
        rmse = [math.sqrt(value / effective_trials) for value in square_bias]
        observed_coverage = [count / effective_trials for count in coverage]
    else:
        mean_estimate = [None] * width
        rmse = [None] * width
        observed_coverage = [None] * width
    clean_trace = _noiseless(rows, amplitudes, candidates, width)
    return {
        "model": "monte-carlo-calibration-v1",
        "trials": trials,
        "effective_trials": effective_trials,
        "unidentifiable_trials": singular,
        "target_channel": target_channel,
        "noise_sigma_per_component": noise_sigma,
        "expected_complex_variance": complex_variance,
        "mean_estimate": [[value.real, value.imag] if value is not None else None
                          for value in mean_estimate],
        "rmse": rmse,
        "coverage_95": observed_coverage,
        "nominal_coverage": 0.95,
        "noiseless_recovery": [[value.real, value.imag] for value in clean_trace],
        "block_size": block_size,
        "seed": seed,
    }
