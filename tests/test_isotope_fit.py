import math
import pytest
from molecular_materials_foundry.api import dispatch
from molecular_materials_foundry.errors import DomainError
from molecular_materials_foundry.isotopes.forward import (
    Instrument, candidate_states, merge_clusters, build_bins, design_column)

POP_X = [[10.0, 0.8], [11.0, 0.2]]
POP_Y = [[1.0, 0.5], [2.0, 0.5]]
INSTRUMENT = {"resolution": 500.0}
BUDGET = {"max_evaluations": 100000, "max_seconds": 30.0}


def simulate(candidate_payloads, fractions, shares, scale, instrument, window=None):
    """Forward-simulate centroid counts through the same bins the fit uses."""
    per_states = [candidate_states(dict(payload, labeling=[
        dict(rule, fraction=fractions[payload["id"]][rule["element"]])
        for rule in payload.get("labeling", [])]), 0, 0.0) for payload in candidate_payloads]
    centroids = [cluster["centroid"] for states in per_states
                 for cluster in merge_clusters(states, instrument)]
    if window is not None:
        centroids = [c for c in centroids if window[0] <= c <= window[1]]
    centers, lower, upper = build_bins([], centroids, window, instrument)
    columns = []
    for states in per_states:
        column, _, _ = design_column(states, centers, lower, upper, window, instrument)
        columns.append(column)
    means = [scale * math.fsum(shares[j] * columns[j][b] for j in range(len(shares)))
             for b in range(len(centers))]
    observations = [{"coordinate": center, "intensity": round(mean)}
                    for center, mean in zip(centers, means) if mean > 0.5]
    return observations, centers, columns


def two_candidates():
    return [
        {"id": "A", "composition": {"X": 4}, "populations": {"X": POP_X},
         "labeling": [{"element": "X", "labeled_isotope": 1, "bounds": [0.0, 1.0]}]},
        {"id": "B", "composition": {"X": 2, "Y": 2},
         "populations": {"X": POP_X, "Y": POP_Y}},
    ]


def test_known_mixture_recovers_shares_fraction_and_scale():
    instrument = Instrument(500.0, None, "constant", "fwhm", 1.0)
    candidates = two_candidates()
    fractions = {"A": {"X": 0.6}, "B": {}}
    observations, _, _ = simulate(candidates, fractions, [0.7, 0.3], 2000.0, instrument)
    result = dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                       "observations": observations, "budget": BUDGET})
    assert result["status"] == "completed"
    shares = {c["id"]: c["share"] for c in result["candidates"]}
    assert shares["A"] == pytest.approx(0.7, abs=0.02)
    assert shares["B"] == pytest.approx(0.3, abs=0.02)
    assert math.fsum(shares.values()) == pytest.approx(1.0)
    label = result["candidates"][0]["labeling"][0]
    assert label["fraction"] == pytest.approx(0.6, abs=0.03)
    assert result["scale"]["value"] == pytest.approx(2000.0, rel=0.02)
    assert result["design"]["rank"] == 2
    # Pearson residuals of a well-specified fit are order one.
    assert result["fit"]["chi_squared"] < result["fit"]["observed_bins"] * 3.0


def test_support_intervals_cover_the_known_truth():
    instrument = Instrument(500.0, None, "constant", "fwhm", 1.0)
    candidates = two_candidates()
    observations, _, _ = simulate(candidates, {"A": {"X": 0.6}, "B": {}},
                                  [0.7, 0.3], 4000.0, instrument)
    result = dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                       "observations": observations, "confidence": 0.9,
                                       "budget": BUDGET})
    interval = result["candidates"][0]["share_interval"]
    assert interval["low"] <= 0.7 <= interval["high"]
    assert not (interval["truncated_low"] or interval["truncated_high"])
    label_interval = result["candidates"][0]["labeling"][0]["fraction_interval"]
    assert label_interval["low"] <= 0.6 <= label_interval["high"]
    scale_interval = result["scale"]["interval"]
    assert scale_interval["low"] <= 4000.0 <= scale_interval["high"]


def test_window_censoring_enters_the_same_caliber_without_renormalization():
    instrument = Instrument(500.0, None, "constant", "fwhm", 1.0)
    candidates = [
        {"id": "A", "composition": {"X": 4}, "populations": {"X": POP_X}},
        {"id": "B", "composition": {"X": 4}, "populations": {"X": POP_X},
         "labeling": [{"element": "X", "labeled_isotope": 1, "bounds": [0.0, 1.0]}]},
    ]
    window = [41.5, 44.5]
    fractions = {"A": {}, "B": {"X": 0.95}}
    observations, _, _ = simulate(candidates, fractions, [0.4, 0.6], 3000.0, instrument, window)
    result = dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                       "observations": observations, "window": window, "budget": BUDGET})
    shares = {c["id"]: c["share"] for c in result["candidates"]}
    assert shares["A"] == pytest.approx(0.4, abs=0.03)
    assert shares["B"] == pytest.approx(0.6, abs=0.03)
    # Most of natural-abundance A sits below the window edge and is not rescaled away.
    assert result["candidates"][0]["probability_inside_window"] < 0.3
    assert result["candidates"][0]["outside_window_probability"] > 0.7
    assert result["error_budget"]["renormalized"] is False
    assert math.fsum(c["probability_inside_window"] + c["outside_window_probability"]
                     for c in result["candidates"]) == pytest.approx(2.0)


def test_identical_candidates_are_reported_as_a_group_not_split():
    instrument = Instrument(500.0, None, "constant", "fwhm", 1.0)
    candidates = [
        {"id": "P", "composition": {"X": 3}, "populations": {"X": POP_X}},
        {"id": "Q", "composition": {"X": 3}, "populations": {"X": POP_X}},
    ]
    observations, _, _ = simulate(candidates, {"P": {}, "Q": {}}, [0.5, 0.5], 1500.0, instrument)
    result = dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                       "observations": observations})
    assert result["design"]["rank"] == 1
    assert result["status"] == "completed_with_limits"
    group = result["indistinguishable_groups"]
    assert len(group) == 1 and group[0]["candidate_ids"] == ["P", "Q"]
    assert group[0]["total_share"] == pytest.approx(1.0)
    # Individual shares are not identifiable, so no individual interval is offered.
    assert all(c["share_interval"] is None for c in result["candidates"])


def test_gaussian_likelihood_uses_errors_and_detection_limit():
    instrument = Instrument(500.0, None, "constant", "fwhm", 1.0)
    candidates = [{"id": "P", "composition": {"X": 3}, "populations": {"X": POP_X}}]
    observations, centers, columns = simulate(
        candidates, {"P": {}}, [1.0], 1500.0, instrument)
    gaussian = [{"coordinate": item["coordinate"],
                 "intensity": item["intensity"],
                 "error": math.sqrt(max(1.0, item["intensity"]))} for item in observations]
    result = dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                       "observations": gaussian, "likelihood": "gaussian",
                                       "detection_limit": 1.0})
    assert result["likelihood"] == "gaussian"
    assert result["candidates"][0]["share"] == pytest.approx(1.0, abs=1e-6)
    assert result["scale"]["interval"] is not None


def test_gaussian_requires_detection_limit_for_unreported_bins():
    candidates = [{"id": "P", "composition": {"X": 3}, "populations": {"X": POP_X}}]
    with pytest.raises(DomainError, match="detection_limit"):
        dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                  "observations": [{"coordinate": 30.0, "intensity": 1.0,
                                                    "error": 0.5}], "likelihood": "gaussian"})


def test_exhausted_budget_is_reported_incomplete_without_crashing():
    instrument = Instrument(500.0, None, "constant", "fwhm", 1.0)
    candidates = two_candidates()
    observations, _, _ = simulate(candidates, {"A": {"X": 0.6}, "B": {}},
                                  [0.7, 0.3], 2000.0, instrument)
    result = dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                       "observations": observations,
                                       "budget": {"max_evaluations": 150, "max_seconds": 30.0}})
    assert result["status"] == "incomplete"
    assert any("limit" in reason for reason in result["status_reasons"])
    assert result["budget"]["evaluations"] >= 150


def test_time_budget_is_honored():
    instrument = Instrument(500.0, None, "constant", "fwhm", 1.0)
    observations, _, _ = simulate(two_candidates(), {"A": {"X": 0.6}, "B": {}},
                                  [0.7, 0.3], 2000.0, instrument)
    result = dispatch("/isotope_fit", {"candidates": two_candidates(), "instrument": INSTRUMENT,
                                       "observations": observations,
                                       "budget": {"max_evaluations": 2000000, "max_seconds": 0.02}})
    assert result["status"] == "incomplete"
    assert any("time_limit" in reason for reason in result["status_reasons"])
    assert result["budget"]["elapsed_seconds"] <= 1.0


def test_zero_count_spectrum_yields_full_range_uncertainty():
    candidates = [{"id": "P", "composition": {"X": 3}, "populations": {"X": POP_X}}]
    result = dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                       "observations": []})
    interval = result["candidates"][0]["share_interval"]
    assert interval["truncated_low"] and interval["truncated_high"]
    assert result["fit"]["observed_bins"] == 0


def test_merged_peak_uncertainty_is_quantified():
    # Low resolution so neighboring isotope states merge into one bin.
    instrument = Instrument(20.0, None, "constant", "fwhm", 1.0)
    candidates = [{"id": "P", "composition": {"X": 6}, "populations": {"X": POP_X}}]
    observations, _, _ = simulate(candidates, {"P": {}}, [1.0], 5000.0, instrument)
    result = dispatch("/isotope_fit", {"candidates": candidates,
                                       "instrument": {"resolution": 20.0},
                                       "observations": observations, "budget": BUDGET})
    assert result["error_budget"]["merge_sigma_mz_max"] > 0.0
    assert any(bin_["merged_state_count"] > 1 for bin_ in result["residuals"])
    assert result["error_budget"]["tail_outside_bins"][0] >= 0.0


def test_charge_and_adduct_shift_every_candidate_coordinate():
    from molecular_materials_foundry.isotopes.populations import ELECTRON_MASS
    instrument = Instrument(2000.0, None, "constant", "fwhm", 1.0)
    candidates = [{"id": "P", "composition": {"X": 2}, "populations": {"X": POP_X}}]
    observations, _, _ = simulate(candidates, {"P": {}}, [1.0], 3000.0, instrument)
    charge, adduct = 2, 1.0
    ionized = [dict(item, coordinate=(item["coordinate"] + adduct - charge * ELECTRON_MASS) / charge)
               for item in observations]
    charged = dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                        "observations": ionized, "charge": charge,
                                        "adduct_mass": adduct})
    assert charged["coordinate"] == "mass_to_charge"
    neutral = dispatch("/isotope_fit", {"candidates": candidates, "instrument": INSTRUMENT,
                                        "observations": observations})
    for neutral_bin, charged_bin in zip(neutral["residuals"], charged["residuals"]):
        expected = (neutral_bin["coordinate"] + adduct - charge * ELECTRON_MASS) / charge
        assert charged_bin["coordinate"] == pytest.approx(expected, abs=1e-12)
    # The ionized spectrum still fits the single candidate cleanly.
    assert charged["candidates"][0]["share"] == pytest.approx(1.0, abs=1e-4)
    assert charged["status"] != "incomplete"


@pytest.mark.parametrize("payload,code", [
    ({"candidates": [], "instrument": INSTRUMENT, "observations": []}, "invalid_array"),
    ({"candidates": [{"id": "P", "composition": {"X": 1}, "populations": {"X": POP_X}}],
      "instrument": {"resolution": 500.0, "scaling": "sqrt_mz"}, "observations": []},
     "missing_resolution_at"),
    ({"candidates": [{"id": "P", "composition": {"X": 1}, "populations": {"X": POP_X}}],
      "instrument": INSTRUMENT, "observations": [], "likelihood": "bayes"}, "invalid_likelihood"),
    ({"candidates": [{"id": "P", "composition": {"X": 1}, "populations": {"X": POP_X},
                      "labeling": [{"element": "Y", "labeled_isotope": 0, "bounds": [0, 1]}]}],
      "instrument": INSTRUMENT, "observations": []}, "labeling_element"),
    ({"candidates": [{"id": "P", "composition": {"X": 1}, "populations": {"X": POP_X}}],
      "instrument": INSTRUMENT, "observations": [{"coordinate": 10.0, "intensity": -1}]},
     "invalid_integer"),
])
def test_invalid_requests_are_rejected(payload, code):
    with pytest.raises(DomainError) as exc:
        dispatch("/isotope_fit", payload)
    assert exc.value.code == code


def test_no_window_conserves_probability_and_quantifies_spill():
    instrument = Instrument(300.0, None, "constant", "fwhm", 1.0)
    candidates = [{"id": "P", "composition": {"X": 5}, "populations": {"X": POP_X}}]
    observations, _, _ = simulate(candidates, {"P": {}}, [1.0], 6000.0, instrument)
    result = dispatch("/isotope_fit", {"candidates": candidates,
                                       "instrument": {"resolution": 300.0},
                                       "observations": observations, "budget": BUDGET})
    record = result["candidates"][0]
    # Without a window all probability belongs to the observation caliber: only
    # the finite Gaussian tail cutoff (6 sigma) may remain outside the bins.
    assert record["probability_inside_window"] == pytest.approx(1.0)
    assert record["tail_outside_bins"] < 1e-6
    column_total = math.fsum(bin_["fitted"] for bin_ in result["residuals"])
    assert column_total == pytest.approx(result["scale"]["value"], rel=1e-6)
    # Broadened neighboring states leak into adjacent bins by a reported amount.
    assert result["error_budget"]["cross_bin_spill"][0] >= 0.0
    assert result["error_budget"]["renormalized"] is False


def test_design_column_partitions_probability_without_renormalization():
    from molecular_materials_foundry.isotopes.forward import design_column
    instrument = Instrument(500.0, None, "constant", "fwhm", 1.0)
    states = candidate_states({"id": "P", "composition": {"X": 6}, "populations": {"X": POP_X}}, 0, 0.0)
    centroids = [c["centroid"] for c in merge_clusters(states, instrument)]
    centers, lower, upper = build_bins([], centroids, None, instrument)
    column, outside, _ = design_column(states, centers, lower, upper, None, instrument)
    total = math.fsum(column) + outside
    assert total == pytest.approx(1.0, abs=1e-9)
    # A window that excludes every state reports the full mass as censored.
    window = [990.0, 1000.0]
    centers_w, lower_w, upper_w = build_bins([], [], window, instrument)
    assert centers_w == []
    column, outside, _ = design_column(states, centers_w, lower_w, upper_w, window, instrument)
    assert column == [] and outside == pytest.approx(1.0)
    with pytest.raises(DomainError, match="theoretical mass"):
        dispatch("/isotope_fit", {"candidates": [{"id": "P", "composition": {"X": 6},
                                                  "populations": {"X": POP_X}}],
                                  "instrument": {"resolution": 300.0},
                                  "observations": [], "window": window})


def test_precise_peak_endpoint_is_unchanged():
    result = dispatch("/isotopes", {"composition": {"X": 2},
                                    "populations": {"X": [[10, 0.8], [11, 0.2]]}})
    assert [p["coordinate"] for p in result["peaks"]] == [20, 21, 22]
    assert result["model"] == "complete-composition-enumeration-v1"
