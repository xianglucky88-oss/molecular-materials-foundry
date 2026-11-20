"""Tests for multi-batch reflection scaling and merging."""
import json
import math
import pytest
from molecular_materials_foundry.api import dispatch
from molecular_materials_foundry.errors import DomainError

IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
INVERSION = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
P21 = [
    {"rotation": IDENTITY, "translation": [0, 0, 0]},
    {"rotation": INVERSION, "translation": [0, 0.5, 0]},
]
COMMON = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 1)]


def _batches_two(obs_a, obs_b, **extra_a):
    batch_a = {"id": "A", "observations": obs_a}
    batch_a.update(extra_a)
    return {"batches": [batch_a, {"id": "B", "observations": obs_b}]}


def _scales(result):
    return {entry["batch"]: entry["scale"]
            for group in result["scale_groups"] for entry in group["batch_scales"]}


def _by_hkl(result):
    return {tuple(ref["hkl"]): ref for ref in result["reflections"]}


def test_known_relative_scale_is_recovered_and_applied_in_anchor_frame():
    obs_a = [{"hkl": list(h), "intensity": 10.0 + k, "variance": 1.0}
             for k, h in enumerate(COMMON)]
    obs_b = [{"hkl": list(h), "intensity": 2.0 * (10.0 + k), "variance": 4.0}
             for k, h in enumerate(COMMON)]
    result = dispatch("/reflections", _batches_two(obs_a, obs_b))
    scales = _scales(result)
    assert scales["A"] == pytest.approx(1.0)
    assert scales["B"] / scales["A"] == pytest.approx(2.0)
    edge = result["scale_groups"][0]["pair_estimates"][0]
    assert edge["overlapping_reflections"] == 4
    assert edge["scale_ratio_j_over_i"] == pytest.approx(2.0)
    assert edge["status"] == "ok"
    # Merged values live in the anchor (A) frame: B readings are divided back.
    by_hkl = _by_hkl(result)
    for k, h in enumerate(COMMON):
        representative = tuple(-x for x in h)
        assert by_hkl[representative]["intensity"] == pytest.approx(10.0 + k)
        # B raw variance 4 with scale 2 -> scaled variance 1; merged with A's 1 -> 0.5.
        assert by_hkl[representative]["variance"] == pytest.approx(0.5)


def test_legal_basis_change_leaves_merged_result_invariant():
    p_matrix = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]  # det +1, b' = b - a
    p_inverse = [[1, 0, 0], [-1, 1, 0], [0, 0, 1]]
    obs_a = [{"hkl": list(h), "intensity": 10.0 + k, "variance": 1.0}
             for k, h in enumerate(COMMON)]
    obs_b = [{"hkl": list(h), "intensity": 2.0 * (10.0 + k), "variance": 4.0}
             for k, h in enumerate(COMMON)]
    baseline = dispatch("/reflections", _batches_two(obs_a, obs_b))

    def to_batch(hkl):
        return [sum(hkl[j] * p_inverse[j][r] for j in range(3)) for r in range(3)]

    obs_b_transformed = [
        {"hkl": to_batch(h), "intensity": 2.0 * (10.0 + k), "variance": 4.0}
        for k, h in enumerate(COMMON)]
    moved = dispatch("/reflections", {"batches": [
        {"id": "A", "observations": obs_a},
        {"id": "B", "basis_change": p_matrix, "observations": obs_b_transformed}]})
    assert _scales(moved)["B"] == pytest.approx(2.0)
    base_reflections = sorted(baseline["reflections"], key=lambda r: r["hkl"])
    moved_reflections = sorted(moved["reflections"], key=lambda r: r["hkl"])
    assert [r["hkl"] for r in base_reflections] == [r["hkl"] for r in moved_reflections]
    for left, right in zip(base_reflections, moved_reflections):
        assert left["intensity"] == pytest.approx(right["intensity"])
        assert left["variance"] == pytest.approx(right["variance"])


def test_batches_without_supported_overlap_are_not_force_merged():
    result = dispatch("/reflections", {"batches": [
        {"id": "A", "observations": [{"hkl": [1, 0, 0], "intensity": 10.0, "variance": 1.0}]},
        {"id": "B", "observations": [{"hkl": [2, 0, 0], "intensity": 20.0, "variance": 1.0}]}]})
    assert len(result["scale_groups"]) == 2
    assert all(group["status"] == "unanchored" for group in result["scale_groups"])
    assert result["unsupported_overlaps"][0]["reason"] == "fewer_than_two_overlapping_reflections"
    assert any(w["code"] == "batch_not_comparable" for w in result["warnings"])
    # Each reflection retains its own scale group and its raw value.
    by_hkl = _by_hkl(result)
    assert by_hkl[(-1, 0, 0)]["intensity"] == pytest.approx(10.0)
    assert by_hkl[(-2, 0, 0)]["intensity"] == pytest.approx(20.0)


def test_single_overlapping_reflection_cannot_anchor_a_scale():
    result = dispatch("/reflections", {"batches": [
        {"id": "A", "observations": [
            {"hkl": [1, 0, 0], "intensity": 10.0, "variance": 1.0},
            {"hkl": [2, 0, 0], "intensity": 11.0, "variance": 1.0}]},
        {"id": "B", "observations": [
            {"hkl": [1, 0, 0], "intensity": 20.0, "variance": 1.0},
            {"hkl": [3, 0, 0], "intensity": 30.0, "variance": 1.0}]}]})
    assert len(result["scale_groups"]) == 2
    assert result["unsupported_overlaps"][0]["overlapping_reflections"] == 1


def test_shared_reflection_across_scale_groups_is_reported_not_force_merged():
    result = dispatch("/reflections", {"batches": [
        {"id": "A", "observations": [
            {"hkl": [1, 0, 0], "intensity": 10.0, "variance": 1.0},
            {"hkl": [2, 0, 0], "intensity": 11.0, "variance": 1.0}]},
        {"id": "B", "observations": [
            {"hkl": [1, 0, 0], "intensity": 40.0, "variance": 1.0},
            {"hkl": [3, 0, 0], "intensity": 30.0, "variance": 1.0}]}]})
    shared = next(r for r in result["reflections"] if r["hkl"] == [-1, 0, 0])
    assert shared["status"] == "scale_groups_not_comparable"
    assert shared["intensity"] is None and shared["variance"] is None
    per_group = {g["anchor_batch"]: g["intensity"]
                 for g in shared["per_scale_group_intensity"]}
    assert per_group == {"A": 10.0, "B": 40.0}
    assert any(w["code"] == "reflection_not_merged_across_scale_groups"
               for w in result["warnings"])
    # Reflections seen by only one group still receive an ordinary estimate.
    only_a = next(r for r in result["reflections"] if r["hkl"] == [-2, 0, 0])
    assert only_a["status"] == "ok" and only_a["intensity"] == pytest.approx(11.0)


def test_friedel_averaged_batch_cannot_anchor_a_preserved_scale():
    # The only overlap is a Friedel-averaged reading; with opposites preserved
    # it cannot support a scale, so the batches stay in separate scale groups.
    result = dispatch("/reflections", {"friedel": False, "batches": [
        {"id": "A", "anomalous": "preserve", "observations": [
            {"hkl": [1, 0, 0], "intensity": 12.0, "variance": 1.0},
            {"hkl": [-1, 0, 0], "intensity": 8.0, "variance": 1.0},
            {"hkl": [2, 0, 0], "intensity": 5.0, "variance": 1.0}]},
        {"id": "B", "anomalous": "merge", "observations": [
            {"hkl": [1, 0, 0], "intensity": 10.0, "variance": 1.0},
            {"hkl": [2, 0, 0], "intensity": 99.0, "variance": 1.0}]}]})
    assert len(result["scale_groups"]) == 2
    assert result["unsupported_overlaps"][0]["reason"] == \
        "fewer_than_two_overlapping_reflections"


def test_systematic_absences_are_distinguished_from_unmeasured_reflections():
    payload = {
        "batches": [{"id": "A", "space_group": P21, "anomalous": "preserve", "observations": [
            {"hkl": [1, 0, 0], "intensity": 9.0, "variance": 1.0},
            {"hkl": [0, 1, 0], "intensity": 8.0, "variance": 1.0},
            {"hkl": [0, 3, 0], "intensity": 0.2, "variance": 1.0}]}],
        "expected_reflections": [[1, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [2, 2, 2]],
    }
    result = dispatch("/reflections", payload)
    absent = {tuple(r["hkl"]) for r in result["systematic_absences"]}
    assert absent == {(0, -1, 0), (0, -3, 0)}
    record = next(r for r in result["systematic_absences"] if r["hkl"] == [0, -1, 0])
    assert record["observed"] is True
    assert record["conflicting_observations"][0]["intensity"] == 8.0
    assert record["extinction_operation"]["translation"] == ["0", "1/2", "0"]
    unmeasured = {tuple(r["hkl"]) for r in result["unmeasured_reflections"]}
    # (0,2,0) is allowed by 2_1 (even k) but not measured; extinct ones never appear here.
    assert unmeasured == {(0, -2, 0), (-2, -2, -2)}
    assert any(w["code"] == "extinction_conflict" for w in result["warnings"])


def test_centering_extinction_survives_friedel_preserved_mode():
    centering = [
        {"rotation": IDENTITY, "translation": [0, 0, 0]},
        {"rotation": IDENTITY, "translation": [0.5, 0.5, 0.5]},
    ]
    payload = {"friedel": False,
               "expected_reflections": [[1, 0, 0]],
               "batches": [{"id": "A", "space_group": centering,
               "anomalous": "preserve",
               "observations": [{"hkl": [1, 1, 0], "intensity": 5.0, "variance": 1.0}]}]}
    result = dispatch("/reflections", payload)
    # h+k+l odd is extinct even when Friedel opposites are reported separately.
    assert any(tuple(r["hkl"]) == (1, 0, 0) for r in result["systematic_absences"])
    # The allowed measured reflection is not reported as absent.
    assert all(r["hkl"] != [1, 1, 0] for r in result["systematic_absences"])


def test_friedel_contrast_is_quantified_when_preserved():
    # Both batches carry the same Friedel pattern; B is recorded at 2x scale.
    payload = {"friedel": False, "batches": [
        {"id": "A", "anomalous": "preserve", "observations": [
            {"hkl": [1, 0, 0], "intensity": 12.0, "variance": 1.0},
            {"hkl": [-1, 0, 0], "intensity": 8.0, "variance": 1.0}]},
        {"id": "B", "anomalous": "preserve", "observations": [
            {"hkl": [1, 0, 0], "intensity": 24.0, "variance": 4.0},
            {"hkl": [-1, 0, 0], "intensity": 16.0, "variance": 4.0}]}]}
    result = dispatch("/reflections", payload)
    plus = next(r for r in result["reflections"] if r["hkl"] == [1, 0, 0])
    minus = next(r for r in result["reflections"] if r["hkl"] == [-1, 0, 0])
    # Back on A's scale the opposites reproduce A's values exactly.
    assert plus["intensity"] == pytest.approx(12.0)
    assert minus["intensity"] == pytest.approx(8.0)
    assert plus["friedel_contrast"]["partner_hkl"] == [-1, 0, 0]
    assert plus["friedel_contrast"]["friedel_delta"] == pytest.approx(4.0)
    assert minus["friedel_contrast"]["friedel_delta"] == pytest.approx(-4.0)


def test_friedel_averaged_observations_cannot_populate_preserved_opposites():
    payload = {"friedel": False, "batches": [
        {"id": "A", "anomalous": "preserve", "observations": [
            {"hkl": [1, 0, 0], "intensity": 12.0, "variance": 1.0},
            {"hkl": [-1, 0, 0], "intensity": 8.0, "variance": 1.0}]},
        {"id": "B", "anomalous": "merge", "observations": [
            {"hkl": [1, 0, 0], "intensity": 10.0, "variance": 1.0}]}]}
    result = dispatch("/reflections", payload)
    hkls = {tuple(r["hkl"]) for r in result["reflections"]}
    assert hkls == {(1, 0, 0), (-1, 0, 0)}
    excluded = [e for e in result["excluded_observations"]
                if e["reason"] == "friedel_averaged_observation_under_preserve_request"]
    assert excluded and excluded[0]["batch"] == "B"
    assert any(w["code"] == "friedel_averaged_observations_excluded" for w in result["warnings"])
    # The preserved opposite carries only batch A's statistics; B's averaged
    # reading is reported but marked excluded so it cannot set either opposite.
    plus = next(r for r in result["reflections"] if r["hkl"] == [1, 0, 0])
    assert plus["intensity"] == pytest.approx(12.0)
    b_contribution = next(c for c in plus["batch_contributions"] if c["batch"] == "B")
    assert all(o["included"] is False for o in b_contribution["observations"])


def test_friedel_default_merges_opposites_without_contrast_field():
    result = dispatch("/reflections", {"batches": [
        {"id": "A", "anomalous": "preserve", "observations": [
            {"hkl": [1, 0, 0], "intensity": 12.0, "variance": 1.0},
            {"hkl": [-1, 0, 0], "intensity": 8.0, "variance": 1.0}]}]})
    assert result["friedel_merged"] is True
    assert len(result["reflections"]) == 1
    ref = result["reflections"][0]
    assert "friedel_contrast" not in ref
    assert ref["intensity"] == pytest.approx(10.0)


def test_negative_intensities_are_retained_and_flagged():
    result = dispatch("/reflections", {"batches": [{"id": "A", "observations": [
        {"hkl": [1, 0, 0], "intensity": -3.0, "variance": 1.0},
        {"hkl": [1, 0, 0], "intensity": -1.0, "variance": 1.0}]}]})
    ref = result["reflections"][0]
    assert ref["status"] == "negative_intensity"
    assert ref["intensity"] == pytest.approx(-2.0)
    assert ref["variance"] == pytest.approx(0.5)


def test_invalid_variances_are_excluded_not_silently_dropped():
    result = dispatch("/reflections", {"batches": [{"id": "A", "observations": [
        {"hkl": [1, 0, 0], "intensity": 7.0, "variance": 1.0},
        {"hkl": [1, 0, 0], "intensity": 99.0, "variance": 0.0},
        {"hkl": [2, 0, 0], "intensity": 5.0, "variance": -2.0}]}]})
    by_hkl = _by_hkl(result)
    valid = by_hkl[(-1, 0, 0)]
    assert valid["intensity"] == pytest.approx(7.0)  # valid observation survives
    invalid = by_hkl[(-2, 0, 0)]
    assert invalid["status"] == "no_valid_variance"
    assert invalid["intensity"] is None and invalid["variance"] is None
    reasons = [e["reason"] for e in result["excluded_observations"]]
    assert reasons.count("non_positive_or_invalid_variance") == 2
    assert any(w["code"] == "invalid_variance" for w in result["warnings"])


def test_outliers_are_flagged_and_kept_in_the_average():
    result = dispatch("/reflections", {"batches": [{"id": "A", "observations": [
        {"hkl": [1, 0, 0], "intensity": 10.0, "variance": 1.0},
        {"hkl": [1, 0, 0], "intensity": 10.1, "variance": 1.0},
        {"hkl": [1, 0, 0], "intensity": 9.9, "variance": 1.0},
        {"hkl": [1, 0, 0], "intensity": 10.2, "variance": 1.0},
        {"hkl": [1, 0, 0], "intensity": 15.0, "variance": 1.0}]}]})
    ref = result["reflections"][0]
    assert "outliers_present" in ref["flags"]
    observations = ref["batch_contributions"][0]["observations"]
    flagged = [o for o in observations if o["outlier"]]
    assert len(flagged) == 1 and flagged[0]["intensity"] == 15.0
    assert all(o["included"] for o in observations)
    # Nothing was deleted: the mean still includes the 15.0 reading.
    assert ref["n_observations"] == 5
    assert ref["intensity"] == pytest.approx(11.04)


def test_scale_outlier_does_not_pull_the_pair_fit():
    obs_c = [{"hkl": list(h), "intensity": 10.0 + k, "variance": 1.0}
             for k, h in enumerate(COMMON)]
    obs_d = [{"hkl": list(h), "intensity": (99.0 if k == 2 else 10.0 + k), "variance": 1.0}
             for k, h in enumerate(COMMON)]
    result = dispatch("/reflections", {"batches": [
        {"id": "C", "observations": obs_c}, {"id": "D", "observations": obs_d}]})
    assert result["scale_outliers"][0]["class"] == [-1, -1, 0]
    edge = result["scale_groups"][0]["pair_estimates"][0]
    assert edge["excluded_classes"] == [[-1, -1, 0]]
    assert edge["scale_ratio_j_over_i"] == pytest.approx(1.0)
    assert edge["reflections_used"] == 3 and edge["overlapping_reflections"] == 4


def test_between_batch_tension_is_localized_to_the_conflicted_reflection():
    obs_a = [{"hkl": list(h), "intensity": 10.0 + k, "variance": 1.0}
             for k, h in enumerate(COMMON)]
    obs_b = [{"hkl": list(h), "intensity": value, "variance": 1.0}
             for h, value in zip(COMMON, (10.0, 11.0, 50.0, 13.0))]
    result = dispatch("/reflections", _batches_two(obs_a, obs_b))
    by_hkl = _by_hkl(result)
    assert "between_batch_tension" in by_hkl[(-1, -1, 0)]["flags"]
    assert "between_batch_tension" not in by_hkl[(-1, 0, 0)]["flags"]


def test_scales_propagate_across_a_chain_with_direct_overlap_absent():
    bc = [(1, 0, 0), (0, 1, 0), (2, 2, 0), (3, 0, 1)]
    obs_a = [{"hkl": list(h), "intensity": 10.0 + k, "variance": 1.0}
             for k, h in enumerate(COMMON)]
    obs_b = ([{"hkl": list(h), "intensity": 2.0 * (10.0 + k), "variance": 4.0}
              for k, h in enumerate(COMMON)]
             + [{"hkl": list(h), "intensity": 2.0 * (10.0 + k), "variance": 4.0}
                for k, h in enumerate(bc)])
    obs_c = [{"hkl": list(h), "intensity": 6.0 * (10.0 + k), "variance": 36.0}
             for k, h in enumerate(bc)]
    result = dispatch("/reflections", {"batches": [
        {"id": "A", "observations": obs_a},
        {"id": "B", "observations": obs_b},
        {"id": "C", "observations": obs_c}]})
    assert len(result["scale_groups"]) == 1
    group = result["scale_groups"][0]
    assert group["cycle_consistency"]["status"] == "ok"
    scales = _scales(result)
    assert scales["B"] / scales["A"] == pytest.approx(2.0)
    assert scales["C"] / scales["B"] == pytest.approx(3.0)


def test_mirror_operation_does_not_erase_friedel_opposites():
    mirror = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    group = [
        {"rotation": IDENTITY, "translation": [0, 0, 0]},
        {"rotation": mirror, "translation": [0, 0, 0]},
    ]
    result = dispatch("/reflections", {"friedel": False, "batches": [
        {"id": "A", "space_group": group, "anomalous": "preserve", "observations": [
            {"hkl": [1, 2, 3], "intensity": 10.0, "variance": 1.0},
            {"hkl": [-1, 2, 3], "intensity": 12.0, "variance": 1.0},
            {"hkl": [-1, -2, -3], "intensity": 5.0, "variance": 1.0}]}]})
    orbits = sorted(tuple(r["hkl"]) for r in result["reflections"])
    # The mirror relates the first two, but the inversion-related opposite stays separate.
    assert orbits == [(-1, -2, -3), (-1, 2, 3)]
    same_orbit = next(r for r in result["reflections"] if r["hkl"] == [-1, 2, 3])
    assert same_orbit["intensity"] == pytest.approx(11.0)
    opposite = next(r for r in result["reflections"] if r["hkl"] == [-1, -2, -3])
    assert opposite["intensity"] == pytest.approx(5.0)


def test_glide_plane_extinction_condition():
    mirror = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    glide = [
        {"rotation": IDENTITY, "translation": [0, 0, 0]},
        {"rotation": mirror, "translation": [0, 0.5, 0]},
    ]
    result = dispatch("/reflections", {
        "friedel": False,
        "expected_reflections": [[0, 1, 0], [0, 2, 0], [1, 1, 0]],
        "batches": [{"id": "A", "space_group": glide, "anomalous": "preserve",
                     "observations": [{"hkl": [1, 1, 0], "intensity": 9.0, "variance": 1.0}]}]})
    assert [r["hkl"] for r in result["systematic_absences"]] == [[0, 1, 0]]
    assert [r["hkl"] for r in result["unmeasured_reflections"]] == [[0, 2, 0]]


def test_space_groups_must_agree_after_basis_change():
    p1 = [{"rotation": IDENTITY, "translation": [0, 0, 0]}]
    with pytest.raises(DomainError, match="same space group"):
        dispatch("/reflections", {"batches": [
            {"id": "A", "space_group": p1, "observations": [
                {"hkl": [1, 0, 0], "intensity": 1.0, "variance": 1.0}]},
            {"id": "B", "space_group": P21, "observations": [
                {"hkl": [1, 0, 0], "intensity": 1.0, "variance": 1.0}]}]})


def test_affine_group_must_be_closed_and_twelfth_valued():
    open_group = [
        {"rotation": IDENTITY, "translation": [0, 0, 0]},
        {"rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "translation": [0.25, 0.25, 0]},
    ]
    with pytest.raises(DomainError) as exc:
        dispatch("/reflections", {"batches": [{"id": "A", "space_group": open_group,
                   "observations": [{"hkl": [1, 0, 0], "intensity": 1.0, "variance": 1.0}]}]})
    assert exc.value.code == "invalid_group"
    fifth = [
        {"rotation": IDENTITY, "translation": [0, 0, 0]},
        {"rotation": INVERSION, "translation": [0, 0.2, 0]},
    ]
    with pytest.raises(DomainError) as exc:
        dispatch("/reflections", {"batches": [{"id": "A", "space_group": fifth,
                   "observations": [{"hkl": [1, 0, 0], "intensity": 1.0, "variance": 1.0}]}]})
    assert exc.value.code == "invalid_translation"


def test_basis_change_must_be_handedness_preserving():
    with pytest.raises(DomainError) as exc:
        dispatch("/reflections", {"batches": [{"id": "A",
                   "basis_change": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   "observations": [{"hkl": [1, 0, 0], "intensity": 1.0, "variance": 1.0}]}]})
    assert exc.value.code == "invalid_basis_change"


def test_batch_request_rejects_unknown_fields_and_bad_conventions():
    with pytest.raises(DomainError) as exc:
        dispatch("/reflections", {"batches": [{"id": "A", "observations": [
            {"hkl": [1, 0, 0], "intensity": 1.0, "variance": 1.0}]}], "bogus": True})
    assert exc.value.code == "unsupported_field"
    with pytest.raises(DomainError) as exc:
        dispatch("/reflections", {"batches": [{"id": "A", "anomalous": "weird",
                   "observations": [{"hkl": [1, 0, 0], "intensity": 1.0, "variance": 1.0}]}]})
    assert exc.value.code == "invalid_anomalous"
    with pytest.raises(DomainError) as exc:
        dispatch("/reflections", {"batches": [
            {"id": "A", "observations": [{"hkl": [1, 0, 0], "intensity": 1.0, "variance": 1.0}]},
            {"id": "A", "observations": [{"hkl": [1, 0, 0], "intensity": 1.0, "variance": 1.0}]}]})
    assert exc.value.code == "duplicate_batch_id"


def test_batch_result_is_json_serializable(sample):
    payload = sample("reflections_batches")
    result = dispatch("/reflections", payload)
    json.dumps(result, allow_nan=False)
    assert result["model"] == "explicit-affine-batch-scaling-v1"


def test_legacy_request_shape_is_unchanged(sample):
    result = dispatch("/reflections", sample("reflections"))
    assert result["model"] == "explicit-point-group-independent-errors-v1"
    assert result["reflections"][0]["intensity"] == pytest.approx(12)
    assert result["systematic_absences_evaluated"] is False
