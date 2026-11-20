import cmath
import json
import math

import pytest
from molecular_materials_foundry.api import dispatch
from molecular_materials_foundry.errors import DomainError
from molecular_materials_foundry.nmr.calibration import calibration_check
from molecular_materials_foundry.nmr.estimation import estimate
from molecular_materials_foundry.nmr.acquisition_schedule import prepare_general


def _cycle_rows(channels, phases, receivers=None, noise=0.0):
    rows = []
    for index, phase in enumerate(phases):
        receiver = 0.0 if receivers is None else receivers[index]
        value = sum(amplitude * cmath.exp(1j * math.radians(channel * phase))
                    for channel, amplitude in channels.items())
        rows.append({"phase_degrees": phase, "receiver_degrees": receiver,
                     "signal": [[value.real, value.imag]]})
    return rows


def test_legacy_contract_is_unchanged_for_complete_cycle(sample):
    result = dispatch("/nmr", sample("nmr"))
    assert result["model"] == "single-axis-uniform-phase-cycle-v1"
    assert set(result) == {"model", "channel", "alias_modulus", "channel_residue",
                           "receiver_convention", "signal", "acquisition_count"}
    assert result["signal"][0] == pytest.approx([2, 0], abs=1e-12)


def test_incomplete_cycle_still_identifies_declared_candidates():
    truth = {0: complex(1.5, -0.7), 1: complex(2.0, -3.0)}
    rows = _cycle_rows(truth, [0, 90, 180])
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1],
                               "plan": {"cycle_size": 4}, "acquisitions": rows})
    assert result["model"] == "general-linear-model-coherence-v1"
    target = result["target"]
    assert target["status"].startswith("identifiable")
    assert complex(*target["signal"][0]) == pytest.approx(truth[1], abs=1e-10)
    assert result["plan"]["grid"]["missing_cycle_steps"] == [3]
    assert result["plan"]["grid"]["complete_cycle"] is False
    assert result["alias_modulus"] is None
    assert result["identifiability"]["rank"] == 2


def test_phase_off_declared_grid_is_rejected():
    rows = _cycle_rows({1: 1 + 0j}, [0, 45, 180])
    with pytest.raises(DomainError, match="plan_inconsistent_with_cycle"):
        dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1],
                          "plan": {"cycle_size": 4}, "acquisitions": rows})


def test_aliased_channels_are_unidentifiable_not_zero_filled():
    rows = [{"phase_degrees": 0, "receiver_degrees": 0, "signal": [[1, 0]]},
            {"phase_degrees": 180, "receiver_degrees": 0, "signal": [[0.5, 0]]}]
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1],
                               "acquisitions": rows})
    assert all(channel["status"] == "unidentifiable"
               for channel in result["candidate_channels"]["channels"])
    assert result["target"]["signal"] is None
    alias_class = result["identifiability"]["alias_classes"][0]
    assert sorted(alias_class["channels"]) == [-1, 1]
    assert result["identifiability"]["full_rank"] is False


def test_states_quadrature_separates_coherence_signs():
    a_plus, a_minus = complex(2, -1), complex(0.5, 0.8)
    acquisitions = []
    for increment in range(2):
        for component, pulse_shift, receiver_shift in (("real", 0, 0),
                                                       ("imaginary", 90, 90)):
            pulse = pulse_shift
            receiver = receiver_shift
            value = (a_plus * cmath.exp(1j * math.radians(pulse)) +
                     a_minus * cmath.exp(-1j * math.radians(pulse)))
            observed = value * cmath.exp(-1j * math.radians(receiver))
            acquisitions.append({"phase_degrees": pulse, "receiver_degrees": receiver,
                                 "increment": increment, "quadrature_component": component,
                                 "signal": [[observed.real, observed.imag]]})
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1],
                               "plan": {"quadrature": "states", "increment_count": 2},
                               "acquisitions": acquisitions})
    assert result["plan"]["quadrature"]["complete_quadrature"] is True
    plus = next(channel for channel in result["candidate_channels"]["channels"]
                if channel["channel"] == 1)
    assert plus["status"].startswith("identifiable")
    assert complex(*plus["signal"][0]) == pytest.approx(a_plus, abs=1e-10)


def test_missing_quadrature_partner_is_reported_not_fabricated():
    a_plus, a_minus = complex(2, -1), complex(0.5, 0.8)
    acquisitions = []
    for increment in range(2):
        for component, pulse_shift, receiver_shift in (("real", 0, 0),
                                                       ("imaginary", 90, 90)):
            if increment == 1 and component == "imaginary":
                continue
            value = (a_plus * cmath.exp(1j * math.radians(pulse_shift)) +
                     a_minus * cmath.exp(-1j * math.radians(pulse_shift)))
            observed = value * cmath.exp(-1j * math.radians(receiver_shift))
            acquisitions.append({"phase_degrees": pulse_shift,
                                 "receiver_degrees": receiver_shift,
                                 "increment": increment, "quadrature_component": component,
                                 "signal": [[observed.real, observed.imag]]})
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1],
                               "plan": {"quadrature": "states", "increment_count": 2},
                               "acquisitions": acquisitions})
    assert result["acquisition_count"] == 3
    assert result["plan"]["quadrature"]["missing_quadrature_components"] == [
        {"increment": 1, "missing_component": "imaginary"}]
    assert result["plan"]["quadrature"]["complete_quadrature"] is False


def test_states_tppi_phase_convention_is_validated():
    a_plus, a_minus = complex(2, -1), complex(0.5, 0.8)
    acquisitions = []
    for increment in range(2):
        for component, extra in (("real", 0), ("imaginary", 90)):
            pulse = 90 * increment + extra
            value = (a_plus * cmath.exp(1j * math.radians(pulse)) +
                     a_minus * cmath.exp(-1j * math.radians(pulse)))
            acquisitions.append({"phase_degrees": pulse, "receiver_degrees": 0,
                                 "increment": increment, "quadrature_component": component,
                                 "signal": [[value.real, value.imag]]})
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1],
                               "plan": {"quadrature": "states_tppi", "increment_count": 2},
                               "acquisitions": acquisitions})
    assert result["plan"]["quadrature"]["phase_convention_valid"] is True
    acquisitions[2]["phase_degrees"] = 30
    broken = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1],
                               "plan": {"quadrature": "states_tppi", "increment_count": 2},
                               "acquisitions": acquisitions})
    assert broken["plan"]["quadrature"]["phase_convention_valid"] is False
    assert broken["plan"]["quadrature"]["phase_violations"]


def test_heteroscedastic_variances_drive_weighted_estimates_and_intervals():
    sigma = [0.1, 0.2, 0.15]
    rows = []
    for index, component_sigma in enumerate(sigma):
        phase = 120 * index
        value = sum(amplitude * cmath.exp(1j * math.radians(channel * phase))
                    for channel, amplitude in {0: complex(1, 0.5), 1: complex(2, -1)}.items())
        rows.append({"phase_degrees": phase, "receiver_degrees": 0,
                     "variance": 2 * component_sigma ** 2,
                     "signal": [[value.real, value.imag]]})
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1],
                               "acquisitions": rows})
    assert result["variance"]["weighted_least_squares"] is True
    target = result["target"]
    assert target["status"] == "identifiable"
    assert target["confidence_interval_95"][0] is not None
    assert target["standard_error"][0] > 0
    # Theory: equal-phasis 3-step design gives component SE sigma/sqrt(3)
    # when every acquisition shares the same variance.
    equal_rows = [{**row, "variance": 0.02} for row in rows]
    equal_result = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1],
                                     "acquisitions": equal_rows})
    assert equal_result["target"]["standard_error"][0] == pytest.approx(
        math.sqrt(0.01 / 3), rel=1e-12)
    effective = result["variance"]["effective_acquisition"]
    assert effective["method"] == "kish_effective_sample_size"
    assert effective["maximum"] <= 3 + 1e-12


def test_repeated_measurements_estimate_variance_and_count_measurements():
    import random
    rng = random.Random(3)
    rows = []
    for index in range(3):
        phase = 120 * index
        repeats = []
        for _ in range(5):
            value = sum(amplitude * cmath.exp(1j * math.radians(channel * phase))
                        for channel, amplitude in {0: complex(1, 0.5),
                                                   1: complex(2, -1)}.items())
            error = complex(rng.gauss(0, 0.1), rng.gauss(0, 0.1))
            repeats.append([[(value + error).real, (value + error).imag]])
        rows.append({"phase_degrees": phase, "receiver_degrees": 0, "signals": repeats})
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1],
                               "acquisitions": rows})
    assert result["total_measurements"] == 15
    assert all(detail["variance_status"] == "estimated_from_repeats"
               for detail in result["variance"]["acquisitions"])
    assert complex(*result["target"]["signal"][0]) == pytest.approx(complex(2, -1), abs=0.1)


def test_saturated_design_without_variance_is_unquantified():
    rows = _cycle_rows({0: complex(1, 0), 1: complex(2, 0)}, [0, 180])
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1],
                               "acquisitions": rows})
    assert result["target"]["status"] == "identifiable_unquantified"
    assert result["target"]["standard_error"][0] is None
    assert result["target"]["confidence_interval_95"][0] is None


def test_ill_conditioned_design_is_flagged_separately_from_unidentifiable():
    # Two nearly identical phases make the design badly conditioned but full rank.
    rows = _cycle_rows({0: complex(1, 0), 1: complex(2, 0)}, [0.0, 0.5])
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1],
                               "acquisitions": rows})
    assert result["identifiability"]["full_rank"] is True
    assert result["identifiability"]["ill_conditioned"] is True
    assert result["target"]["status"].endswith("ill_conditioned")


def test_blocked_estimation_is_independent_of_block_size():
    width = 300
    channels = {0: complex(1, 0.5), 1: complex(2, -1)}
    rows = []
    for index in range(3):
        phase = 120 * index
        trace = []
        for time in range(width):
            decay = math.exp(-time / 150.0)
            value = sum(amplitude * decay * cmath.exp(1j * math.radians(channel * phase))
                        for channel, amplitude in channels.items())
            trace.append([value.real, value.imag])
        rows.append({"phase_degrees": phase, "receiver_degrees": 0,
                     "variance": 0.02, "signal": trace})
    model = prepare_general({"channel": 1, "candidate_channels": [0, 1], "acquisitions": rows})
    small = estimate(model["basis"], model["rows"], model["candidates"], block_size=7)
    large = estimate(model["basis"], model["rows"], model["candidates"], block_size=1024)
    index = model["candidates"].index(1)
    assert small["records"][index]["estimate"] == pytest.approx(
        large["records"][index]["estimate"], abs=1e-14)


def test_calibration_recovers_truth_and_nominal_coverage():
    rows = [{"phase": 120 * k, "receiver": 0.0, "mean": [0j]} for k in range(3)]
    amplitudes = [[complex(1.0, 0.5)], [complex(2.0, -1.0)]]
    result = calibration_check(rows, [0, 1], amplitudes, noise_sigma=0.1,
                               trials=1500, seed=42, target_channel=1)
    assert result["unidentifiable_trials"] == 0
    assert abs(result["coverage_95"][0] - 0.95) < 0.04
    assert complex(*result["mean_estimate"][0]) == pytest.approx(amplitudes[1][0], abs=0.03)
    assert result["rmse"][0] == pytest.approx(
        math.sqrt(2 * 0.1 ** 2 / 3), rel=0.1)


def test_calibration_detects_unidentifiable_schedule():
    rows = [{"phase": 0.0, "receiver": 0.0, "mean": [0j]},
            {"phase": 180.0, "receiver": 0.0, "mean": [0j]}]
    amplitudes = [[complex(1, 0)], [complex(2, 0)]]
    result = calibration_check(rows, [-1, 1], amplitudes, noise_sigma=0.1,
                               trials=20, seed=42, target_channel=1)
    assert result["effective_trials"] == 0
    assert result["unidentifiable_trials"] == 20


def test_general_contract_rejects_unknown_and_malformed_fields():
    rows = _cycle_rows({1: 1 + 0j}, [0, 120])
    with pytest.raises(DomainError, match="Unsupported fields"):
        dispatch("/nmr", {"channel": 1, "acquisitions": rows, "bogus": True})
    bad_signal = [{"phase_degrees": 0, "receiver_degrees": 0, "signal": [[1, 0]]},
                  {"phase_degrees": 120, "receiver_degrees": 0,
                   "signal": [[1, 0]], "signals": [[[1, 0]]]}]
    with pytest.raises(DomainError, match="signal or signals"):
        dispatch("/nmr", {"channel": 1, "acquisitions": bad_signal})
    bad_variance = [{"phase_degrees": 0, "receiver_degrees": 0,
                     "variance": -1, "signal": [[1, 0]]},
                    {"phase_degrees": 120, "receiver_degrees": 0, "signal": [[1, 0]]}]
    with pytest.raises(DomainError, match="positive"):
        dispatch("/nmr", {"channel": 1, "acquisitions": bad_variance})


def test_target_is_appended_to_candidate_set_and_flagged():
    rows = _cycle_rows({0: complex(1, 0), 1: complex(2, 0)}, [0, 120, 240])
    result = dispatch("/nmr", {"channel": 1, "candidate_channels": [0],
                               "acquisitions": rows})
    assert result["candidate_channels"]["target_appended"] is True
    assert {channel["channel"] for channel in result["candidate_channels"]["channels"]} == {0, 1}


def test_quantitative_example_is_serializable_and_complete(sample):
    payload = sample("nmr_quantitative")
    result = dispatch("/nmr", payload)
    json.dumps(result, allow_nan=False)
    assert result["model"] == "general-linear-model-coherence-v1"
    assert result["plan"]["grid"]["missing_cycle_steps"] == [3]
