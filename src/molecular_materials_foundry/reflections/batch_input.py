"""Parse multi-batch reflection requests and group observations into orbits."""
from fractions import Fraction

from ..validation import fields, sequence, integer, number, label
from ..errors import DomainError
from .affine_group import (
    IDENTITY,
    parse_operation,
    validate_affine_group,
    parse_basis_change,
    transform_index,
)

ANOMALOUS_CONVENTIONS = ("preserve", "merge")
MAX_BATCHES = 50
MAX_OBSERVATIONS = 10000
MAX_EXPECTED = 10000
ZERO_TRANSLATION = (Fraction(0),) * 3


def parse_batch(raw, batch_index):
    fields(raw, ["id", "observations"],
           ["basis_change", "space_group", "anomalous"])
    batch_id = label(raw["id"], "batch id")
    basis_change = (parse_basis_change(raw["basis_change"]) if "basis_change" in raw
                    else IDENTITY)
    if "space_group" in raw:
        operations = [parse_operation(entry) for entry in
                      sequence(raw["space_group"], "space_group", minimum=1, maximum=192)]
        group = validate_affine_group(operations)
    else:
        group = frozenset({(IDENTITY, ZERO_TRANSLATION)})
    anomalous = raw.get("anomalous", "preserve")
    if anomalous not in ANOMALOUS_CONVENTIONS:
        raise DomainError("invalid_anomalous",
                         "anomalous must be 'preserve' or 'merge'")
    observations = []
    for obs_index, entry in enumerate(sequence(raw["observations"], "observations",
                                               minimum=0, maximum=MAX_OBSERVATIONS)):
        fields(entry, ["hkl", "intensity", "variance"], [])
        hkl = tuple(integer(x, "hkl", minimum=-100000, maximum=100000)
                    for x in sequence(entry["hkl"], "hkl", minimum=3, maximum=3))
        if hkl == (0, 0, 0):
            raise DomainError("direct_beam", "The direct beam is not a reflection")
        intensity = number(entry["intensity"], "intensity", minimum=-1e100, maximum=1e100)
        variance_probe = number(entry["variance"], "variance", minimum=-1e100, maximum=1e100)
        valid_variance = variance_probe >= 1e-100
        observations.append({
            "batch": batch_index,
            "batch_id": batch_id,
            "local_index": obs_index,
            "hkl_batch": hkl,
            "hkl_reference": transform_index(hkl, basis_change),
            "intensity": intensity,
            "variance": variance_probe if valid_variance else None,
            "variance_valid": valid_variance,
            "anomalous": anomalous,
        })
    return {
        "id": batch_id,
        "index": batch_index,
        "basis_change": basis_change,
        "group": group,
        "anomalous": anomalous,
        "observations": observations,
    }


def prepare(payload):
    fields(payload, ["batches"], ["friedel", "expected_reflections"])
    if "friedel" in payload and not isinstance(payload["friedel"], bool):
        raise DomainError("invalid_friedel", "friedel must be a boolean")
    friedel = payload.get("friedel", True)
    batches_raw = sequence(payload["batches"], "batches", minimum=1, maximum=MAX_BATCHES)
    batches = [parse_batch(raw, index) for index, raw in enumerate(batches_raw)]
    if len({batch["id"] for batch in batches}) != len(batches):
        raise DomainError("duplicate_batch_id", "Batch ids must be unique")
    total = sum(len(batch["observations"]) for batch in batches)
    if total == 0:
        raise DomainError("no_observations", "At least one observation is required")
    if total > MAX_OBSERVATIONS:
        raise DomainError("too_many_observations", "At most 10000 observations are supported")
    expected = None
    if "expected_reflections" in payload:
        expected = []
        for raw_hkl in sequence(payload["expected_reflections"], "expected_reflections",
                                minimum=1, maximum=MAX_EXPECTED):
            hkl = tuple(integer(x, "expected hkl", minimum=-100000, maximum=100000)
                        for x in sequence(raw_hkl, "hkl", minimum=3, maximum=3))
            if hkl == (0, 0, 0):
                raise DomainError("direct_beam", "The direct beam is not a reflection")
            expected.append(hkl)
    return batches, friedel, expected


def point_orbit(hkl, rotations, friedel):
    orbit = {tuple(sum(row[j] * hkl[j] for j in range(3)) for row in rotation)
             for rotation in rotations}
    if friedel:
        orbit |= {tuple(-x for x in member) for member in list(orbit)}
    return orbit


def orbit_representative(hkl, rotations, friedel):
    return min(point_orbit(hkl, rotations, friedel))
