import math
import pytest
from molecular_materials_foundry.api import dispatch
from molecular_materials_foundry.errors import DomainError

CUBIC = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
SHEAR = [[10.0, 0.0, 0.0], [1.0, 8.0, 0.0], [0.0, 0.0, 12.0]]


def wrap(fractional):
    image = [int(math.floor(v)) for v in fractional]
    return [v - i for v, i in zip(fractional, image)], image


def known_motion_payload(reference="laboratory"):
    # atom a moves (2, 0, 0) A per step; atom b is stationary
    truth_a = [[0.1, 0.1, 0.1], [2.1, 0.1, 0.1], [4.1, 0.1, 0.1], [6.1, 0.1, 0.1]]
    truth_b = [[0.5, 0.5, 0.5]] * 4
    frames = []
    for ca, cb in zip(truth_a, truth_b):
        wa, ia = wrap([v / 10.0 for v in ca])
        wb, ib = wrap(cb)
        frames.append({"positions": [wa, wb], "images": [ia, ib]})
    return {
        "atoms": ["a", "b"],
        "groups": [{"id": "g1", "atoms": ["a"]}, {"id": "g2", "atoms": ["b"]}],
        "times": [0.0, 1.0, 2.0, 3.0],
        "cells": [CUBIC] * 4,
        "frames": frames,
        "lag_time": 2.0,
        "max_step": 3.0,
        "reference": reference,
    }


def test_legacy_contract_is_untouched(sample):
    result = dispatch("/trajectories", sample("trajectories"))
    assert result["model"] == "fixed-orthorhombic-cell-v1"
    assert [list(v) for v in result["continuous_positions"][1]] == [[11, 1, 1]]
    assert result["windows"][0]["mean_squared_displacement"] == 4
    assert result["lag_frames"] == 1


def test_legacy_rejects_extended_only_fields(sample):
    payload = sample("trajectories")
    payload["groups"] = []
    with pytest.raises(DomainError, match="Unsupported fields"):
        dispatch("/trajectories", payload)


def test_explicit_images_recover_known_motion_by_group():
    result = dispatch("/trajectories", known_motion_payload())
    assert result["model"] == "triclinic-variable-cell-v1"
    assert result["image_mode"] == "explicit"
    for frame, expected_x in enumerate((0.1, 2.1, 4.1, 6.1)):
        assert result["continuous_positions"][frame][0][0] == pytest.approx(expected_x)
    assert len(result["windows"]) == 2
    g1, g2 = result["group_statistics"]
    assert g1["identified_samples"] == 2 and g1["candidate_samples"] == 2
    assert g1["missing_rate"] == 0.0
    assert g1["mean_squared_displacement"] == pytest.approx(16.0)
    assert g2["mean_squared_displacement"] == 0.0
    # the two lag-2 windows overlap, so they are not two independent samples
    assert g1["independent_window_count"] == 1
    assert g1["effective_sample_size"] == 1
    assert result["time_sampling"]["uniform"] is True


def test_affine_cell_deformation_is_not_particle_motion():
    cells = [
        [[10.0, 0, 0], [0, 10, 0], [0, 0, 10]],
        [[11.0, 0, 0], [0, 10, 0], [0, 0, 10]],
        [[12.0, 0, 0], [0, 10, 0], [0, 0, 10]],
    ]
    base = {
        "atoms": ["a"], "groups": [{"id": "g", "atoms": ["a"]}],
        "times": [0.0, 1.0, 2.0], "cells": cells,
        "frames": [{"positions": [[0.1, 0.1, 0.1]], "images": [[0, 0, 0]]} for _ in range(3)],
        "lag_time": 2.0, "max_step": 5.0,
    }
    lab = dispatch("/trajectories", {**base, "reference": "laboratory"})
    aff = dispatch("/trajectories", {**base, "reference": "affine_removed"})
    assert lab["windows"][0]["squared_displacements"][0] == pytest.approx(0.04)
    assert aff["windows"][0]["squared_displacements"][0] == 0.0


def test_nonuniform_times_match_nearest_lag_and_report_unmatched():
    payload = known_motion_payload()
    payload["times"] = [0.0, 1.0, 1.5, 3.0]
    payload["lag_tolerance"] = 0.6
    result = dispatch("/trajectories", payload)
    assert result["time_sampling"]["uniform"] is False
    assert len(result["windows"]) == 3
    assert result["lag_matching"]["unmatched_frame_starts"] == [3]
    assert [w["elapsed_time"] for w in result["windows"]] == [1.5, 2.0, 1.5]


def test_equidistant_lag_match_is_rejected():
    payload = known_motion_payload()
    payload["times"] = [0.0, 0.5, 1.0, 1.5]
    payload["lag_time"] = 0.75
    payload["lag_tolerance"] = 0.3
    with pytest.raises(DomainError) as exc:
        dispatch("/trajectories", payload)
    assert exc.value.code == "ambiguous_lag_match"


def test_missing_frames_contaminate_every_spanning_window():
    payload = known_motion_payload()
    payload["lag_time"] = 1.0
    payload["frames"][2] = {"positions": None}
    result = dispatch("/trajectories", payload)
    statuses = [[s["valid"] for s in w["sample_status"]] for w in result["windows"]]
    assert statuses[0] == [True, True]
    assert statuses[1] == [False, False]  # missing endpoint
    assert statuses[2] == [False, False]  # window spans the gap
    g1 = result["group_statistics"][0]
    assert g1["missing_rate"] == pytest.approx(2 / 3)
    assert g1["anomalous_intervals_count"] == 2
    assert g1["anomalous_intervals"][0]["reason"] == "missing_frame"
    # unidentifiable samples never shrink the reported envelope interval
    assert g1["msd_conservative_envelope"][1] > g1["mean_squared_displacement"]


def test_multiple_image_candidates_are_reported_not_guessed():
    payload = {
        "atoms": ["a"], "groups": [{"id": "g", "atoms": ["a"]}],
        "times": [0.0, 1.0, 2.0], "cells": [CUBIC] * 3,
        "frames": [{"positions": [[0.95, 0.1, 0.1]]},
                   {"positions": [[0.05, 0.1, 0.1]]},
                   {"positions": [[0.10, 0.1, 0.1]]}],
        "lag_time": 1.0, "max_step": 9.0,
    }
    result = dispatch("/trajectories", payload)
    assert result["windows"][0]["sample_status"][0] == \
        {"valid": False, "reason": "ambiguous_image"}
    assert result["windows"][1]["sample_status"][0]["valid"] is True
    group = result["group_statistics"][0]
    assert group["missing_rate"] == 0.5
    assert group["msd_conservative_envelope"] is not None


def test_no_image_satisfying_bound_is_hard_rejection():
    payload = {
        "atoms": ["a"], "groups": [{"id": "g", "atoms": ["a"]}],
        "times": [0.0, 1.0], "cells": [CUBIC] * 2,
        "frames": [{"positions": [[0.95, 0.1, 0.1]]},
                   {"positions": [[0.05, 0.1, 0.1]]}],
        "lag_time": 1.0, "max_step": 0.5,
    }
    with pytest.raises(DomainError) as exc:
        dispatch("/trajectories", payload)
    assert exc.value.code == "step_bound_violation"


def test_explicit_bond_conflict_propagates_as_invalid_sample():
    payload = {
        "atoms": ["a", "b"], "groups": [{"id": "g", "atoms": ["a", "b"]}],
        "times": [0.0, 1.0], "cells": [CUBIC] * 2,
        "frames": [
            {"positions": [[0.95, 0, 0], [0.05, 0, 0]], "images": [[0, 0, 0], [1, 0, 0]]},
            {"positions": [[0.95, 0, 0], [0.05, 0, 0]], "images": [[0, 0, 0], [0, 0, 0]]},
        ],
        "lag_time": 1.0, "max_step": 11.0,
        "bonds": [["a", "b"]], "max_bond_length": 1.1,
    }
    result = dispatch("/trajectories", payload)
    assert all(not s["valid"] for s in result["windows"][0]["sample_status"])
    assert result["windows"][0]["sample_status"][0]["reason"] == "bond_conflict"


def test_bond_closure_resolves_joint_image_that_is_otherwise_ambiguous():
    payload = {
        "atoms": ["a", "b"], "groups": [{"id": "g", "atoms": ["a", "b"]}],
        "times": [0.0, 1.0], "cells": [CUBIC] * 2,
        "frames": [
            {"positions": [[0.95, 0, 0], [0.05, 0, 0]]},
            {"positions": [[0.05, 0, 0], [0.15, 0, 0]]},
        ],
        "lag_time": 1.0, "max_step": 9.5,
        "bonds": [["a", "b"]], "max_bond_length": 1.1,
    }
    bonded = dispatch("/trajectories", payload)
    assert all(s["valid"] for s in bonded["windows"][0]["sample_status"])
    unbonded = dispatch("/trajectories", {k: v for k, v in payload.items()
                                          if k not in ("bonds", "max_bond_length")})
    assert not unbonded["windows"][0]["sample_status"][0]["valid"]


def test_triclinic_shear_explicit_and_bounded_inference_agree():
    path = [[0.9, 0.95, 0.5], [0.9, 0.15, 0.5], [0.9, 0.35, 0.5]]
    inferred_frames = []
    explicit_frames = []
    for index, fractional in enumerate(path):
        wrapped, _ = wrap(fractional)
        inferred_frames.append({"positions": [wrapped]})
        image = [0, 1, 0] if index else [0, 0, 0]
        explicit_frames.append({"positions": [wrapped], "images": [image]})
    common = {"atoms": ["a"], "groups": [{"id": "g", "atoms": ["a"]}],
              "times": [0.0, 1.0, 2.0], "cells": [SHEAR] * 3,
              "lag_time": 1.0, "max_step": 4.0}
    explicit = dispatch("/trajectories", {**common, "frames": explicit_frames})
    inferred = dispatch("/trajectories", {**common, "frames": inferred_frames})
    for f in range(3):
        for k in range(3):
            assert inferred["continuous_positions"][f][0][k] == \
                pytest.approx(explicit["continuous_positions"][f][0][k])
    step = [b - a for a, b in zip(explicit["continuous_positions"][0][0],
                                  explicit["continuous_positions"][1][0])]
    assert step == pytest.approx([0.0, 1.6, 0.0])
    loose = dispatch("/trajectories", {**common, "max_step": 9.0,
                                       "frames": inferred_frames})
    assert not loose["windows"][0]["sample_status"][0]["valid"]
    assert not loose["windows"][0]["sample_status"][0]["valid"]


@pytest.mark.parametrize("mutation,code", [
    (lambda p: p.update(groups=[{"id": "g", "atoms": ["a"]}]), "group_partition"),
    (lambda p: p.update(times=[0, 1]), "invalid_array"),
    (lambda p: p.update(times=[0, 1, 1, 3]), "invalid_time"),
    (lambda p: p.update(cells=[[[0, 0, 0], [0, 10, 0], [0, 0, 10]]] * 4), "degenerate_cell"),
    (lambda p: p.update(reference="guess"), "invalid_reference"),
])
def test_extended_validation_failures(mutation, code):
    payload = known_motion_payload()
    mutation(payload)
    with pytest.raises(DomainError) as exc:
        dispatch("/trajectories", payload)
    assert exc.value.code == code


def test_extended_requires_group_partition_and_wrapped_fractions():
    payload = known_motion_payload()
    payload["frames"][0]["positions"][0][0] = 1.2
    with pytest.raises(DomainError) as exc:
        dispatch("/trajectories", payload)
    assert exc.value.code == "unwrapped_input"
    payload = known_motion_payload()
    payload["frames"][1] = {"positions": payload["frames"][1]["positions"]}
    with pytest.raises(DomainError) as exc:
        dispatch("/trajectories", payload)
    assert exc.value.code == "mixed_image_mode"
