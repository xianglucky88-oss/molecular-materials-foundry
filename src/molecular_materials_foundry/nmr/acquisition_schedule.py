"""Request preparation for single-axis coherence-channel experiments.

Two contract levels share the same physical convention.  The legacy
contract is one complete, uniformly spaced single-axis phase cycle and is
kept exactly as originally specified.  The general contract allows missing
cycle steps, States / States-TPPI quadrature declarations, repeated
measurements and explicit measurement variances; it never manufactures
data for an acquisition that was not made.
"""
import cmath
import math

from ..validation import fields, number, integer, sequence, label
from ..errors import DomainError
from .quadrature import validate_quadrature

ANGLE_TOLERANCE = 1e-6
MAX_ACQUISITIONS = 256
MAX_REPEATS = 64
MAX_SIGNAL_WIDTH = 10000
QUADRATURE_MODES = ("none", "states", "states_tppi")


def _angle_difference(left, right):
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _complex_samples(signal, name):
    samples = sequence(signal, name, minimum=1, maximum=MAX_SIGNAL_WIDTH)
    values = []
    for sample in samples:
        sample = sequence(sample, "complex sample", minimum=2, maximum=2)
        values.append(complex(number(sample[0], "real", minimum=-1e100, maximum=1e100),
                              number(sample[1], "imaginary", minimum=-1e100, maximum=1e100)))
    return values


def _variance_array(variance, width, name="variance"):
    if isinstance(variance, list):
        entries = sequence(variance, name, minimum=1, maximum=width)
        if len(entries) not in (1, width):
            raise DomainError("invalid_variance",
                              "Per-acquisition variance must be one value or one value per signal sample")
        return [number(entry, name, positive=True) for entry in entries]
    return [number(variance, name, positive=True)]


def is_legacy(payload):
    if set(payload) != {"channel", "acquisitions"}:
        return False
    return all(set(acquisition) == {"phase_degrees", "receiver_degrees", "signal"}
               for acquisition in payload["acquisitions"])


def prepare_legacy(payload):
    fields(payload, ["acquisitions", "channel"])
    channel = integer(payload["channel"], "channel", minimum=-100, maximum=100)
    acquisitions = sequence(payload["acquisitions"], "acquisitions", minimum=2, maximum=64)
    phases, receivers, signals = [], [], []
    width = None
    for acquisition in acquisitions:
        fields(acquisition, ["phase_degrees", "receiver_degrees", "signal"])
        phases.append(number(acquisition["phase_degrees"], "phase", minimum=-36000, maximum=36000) % 360)
        receivers.append(number(acquisition["receiver_degrees"], "receiver",
                                minimum=-36000, maximum=36000) % 360)
        values = _complex_samples(acquisition["signal"], "signal")
        if width is not None and width != len(values):
            raise DomainError("unaligned_acquisitions", "All acquisitions must have equal signal length")
        width = len(values)
        signals.append(values)
    size = len(phases)
    expected = [(phases[0] + 360 * i / size) % 360 for i in range(size)]
    if any(abs(a - b) > 1e-7 for a, b in zip(sorted(expected), sorted(phases))):
        raise DomainError("unsupported_schedule", "Use one complete, uniformly spaced single-axis phase cycle")
    return channel, phases, receivers, signals


def _parse_acquisition(acquisition, quadrature_mode):
    required = ["phase_degrees", "receiver_degrees"]
    optional = ["signal", "signals", "variance", "increment", "quadrature_component"]
    if quadrature_mode != "none":
        required += ["increment", "quadrature_component"]
    fields(acquisition, required, optional)
    if "signal" not in acquisition and "signals" not in acquisition:
        raise DomainError("missing_field", "Each acquisition requires signal or signals")
    if "signal" in acquisition and "signals" in acquisition:
        raise DomainError("invalid_acquisition", "Provide either signal or signals, not both")
    phase = number(acquisition["phase_degrees"], "phase", minimum=-36000, maximum=36000) % 360
    receiver = number(acquisition["receiver_degrees"], "receiver",
                      minimum=-36000, maximum=36000) % 360
    if "signals" in acquisition:
        repeats = sequence(acquisition["signals"], "signals", minimum=2, maximum=MAX_REPEATS)
        measurements = [_complex_samples(repeat, "signals entry") for repeat in repeats]
        width = len(measurements[0])
        if any(len(values) != width for values in measurements):
            raise DomainError("unaligned_acquisitions", "Repeated signals must have equal length")
    else:
        measurements = [_complex_samples(acquisition["signal"], "signal")]
        width = len(measurements[0])
    increment = None
    component = None
    if quadrature_mode != "none":
        increment = integer(acquisition["increment"], "increment", minimum=0, maximum=100000)
        component = label(acquisition["quadrature_component"], "quadrature_component")
        if component not in ("real", "imaginary"):
            raise DomainError("invalid_quadrature", 'quadrature_component must be "real" or "imaginary"')
    explicit_variance = None
    if "variance" in acquisition:
        explicit_variance = _variance_array(acquisition["variance"], width)
    return {"phase": phase, "receiver": receiver, "measurements": measurements, "width": width,
            "increment": increment, "component": component, "explicit_variance": explicit_variance}


def _repeat_statistics(measurements, explicit_variance):
    count = len(measurements)
    width = len(measurements[0])
    means = []
    empirical = []
    for column in range(width):
        real_mean = math.fsum(values[column].real for values in measurements) / count
        imag_mean = math.fsum(values[column].imag for values in measurements) / count
        mean = complex(real_mean, imag_mean)
        means.append(mean)
        if count > 1:
            square = math.fsum(abs(values[column] - mean) ** 2 for values in measurements)
            # Sample complex variance sum/(n-1); mean variance divides once more by n.
            empirical.append(square / (count - 1) / count)
        else:
            empirical.append(None)
    if explicit_variance is not None:
        variance = explicit_variance[:]
        status = "known"
    elif count > 1:
        variance = empirical
        status = "estimated_from_repeats"
    else:
        variance = None
        status = "unknown"
    return means, variance, status, empirical, count


def _grid_summary(phases, cycle_size):
    reference = min(phases)
    grid = [(reference + 360.0 * index / cycle_size) % 360 for index in range(cycle_size)]
    covered = set()
    for phase in phases:
        matches = [index for index, grid_phase in enumerate(grid)
                   if _angle_difference(phase, grid_phase) <= ANGLE_TOLERANCE]
        if not matches:
            raise DomainError("plan_inconsistent_with_cycle",
                              "An observed phase is not on the declared uniform cycle grid")
        covered.add(matches[0])
    missing = [index for index in range(cycle_size) if index not in covered]
    return {"cycle_size": cycle_size, "reference_phase_degrees": reference,
            "missing_cycle_steps": missing, "complete_cycle": not missing}


def prepare_general(payload):
    fields(payload, ["channel", "acquisitions"],
           ["candidate_channels", "plan"])
    channel = integer(payload["channel"], "channel", minimum=-100, maximum=100)
    plan = payload.get("plan", {})
    fields(plan, [], ["cycle_size", "increment_count", "quadrature"])
    quadrature_mode = plan.get("quadrature", "none")
    if quadrature_mode not in QUADRATURE_MODES:
        raise DomainError("invalid_quadrature",
                          "plan.quadrature must be one of " + ", ".join(QUADRATURE_MODES))
    cycle_size = None
    if "cycle_size" in plan:
        cycle_size = integer(plan["cycle_size"], "cycle_size", minimum=2, maximum=1024)
    increment_count = None
    if "increment_count" in plan:
        increment_count = integer(plan["increment_count"], "increment_count", minimum=1, maximum=100000)
    acquisitions = sequence(payload["acquisitions"], "acquisitions", minimum=1, maximum=MAX_ACQUISITIONS)
    parsed = [_parse_acquisition(acquisition, quadrature_mode) for acquisition in acquisitions]
    width = parsed[0]["width"]
    if any(row["width"] != width for row in parsed):
        raise DomainError("unaligned_acquisitions", "All acquisitions must have equal signal length")

    candidate_channels = payload.get("candidate_channels")
    target_added = False
    if candidate_channels is None:
        candidates = [channel]
        candidate_source = "target_only"
    else:
        entries = sequence(candidate_channels, "candidate_channels", minimum=1, maximum=64)
        candidates = []
        for entry in entries:
            value = integer(entry, "candidate_channels entry", minimum=-100, maximum=100)
            if value not in candidates:
                candidates.append(value)
        if channel not in candidates:
            candidates.append(channel)
            target_added = True
        candidate_source = "declared"

    quadrature = validate_quadrature(parsed, quadrature_mode, increment_count, ANGLE_TOLERANCE)

    rows = []
    total_repeats = 0
    for row in parsed:
        means, variance, status, _empirical, repeat_count = _repeat_statistics(
            row["measurements"], row["explicit_variance"])
        total_repeats += repeat_count
        rows.append({"phase": row["phase"], "receiver": row["receiver"], "mean": means,
                     "variance": variance, "variance_status": status,
                     "repeat_count": repeat_count,
                     "increment": row["increment"], "component": row["component"]})

    basis = [[cmath.exp(1j * math.radians(q * row["phase"])) for q in candidates] for row in rows]
    phases = [row["phase"] for row in rows]
    grid = _grid_summary(phases, cycle_size) if cycle_size is not None else None
    return {"channel": channel, "candidates": candidates, "candidate_source": candidate_source,
            "target_added_to_candidates": target_added, "rows": rows, "basis": basis,
            "width": width, "plan": {"cycle_size": cycle_size, "increment_count": increment_count,
                                     "quadrature": quadrature_mode,
                                     "grid": grid, "details": quadrature},
            "acquisition_count": len(rows), "total_measurements": total_repeats}


def prepare(payload):
    if is_legacy(payload):
        return ("legacy",) + prepare_legacy(payload)
    return "general", prepare_general(payload)
