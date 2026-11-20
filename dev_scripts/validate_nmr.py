"""Scientific validation of the general /nmr pipeline (developer scratch)."""
import cmath
import json
import math
import random
import sys

sys.path.insert(0, "src")
from molecular_materials_foundry.api import dispatch
from molecular_materials_foundry.errors import DomainError
from molecular_materials_foundry.nmr.calibration import calibration_check
from molecular_materials_foundry.nmr.estimation import estimate

failures = []


def check(name, condition, detail=""):
    print(("PASS" if condition else "FAIL"), name, detail)
    if not condition:
        failures.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# 1. Incomplete 3-of-4 cycle, candidate set {0,1}: target 1 still identifiable.
rng = random.Random(1)
truth = {0: complex(1.5, -0.7), 1: complex(2.0, -3.0)}
rows = []
for k in (0, 1, 2):  # missing the 270 step
    phase = 90 * k
    value = sum(a * cmath.exp(1j * math.radians(q * phase)) for q, a in truth.items())
    rows.append({"phase_degrees": phase, "receiver_degrees": 0,
                 "signal": [[value.real, value.imag]]})
r = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1],
                      "plan": {"cycle_size": 4}, "acquisitions": rows})
check("incomplete cycle target estimated", r["target"]["status"].startswith("identifiable"))
got = complex(*r["target"]["signal"][0])
check("incomplete cycle point estimate", approx(got, truth[1], 1e-10), str(got))
check("missing step reported", r["plan"]["grid"]["missing_cycle_steps"] == [3],
      str(r["plan"]["grid"]))
check("no alias modulus for incomplete cycle", r["alias_modulus"] is None)

# 2. Aliased pair without quadrature: single scan cannot separate +1/-1.
rows2 = [{"phase_degrees": 0, "receiver_degrees": 0, "signal": [[1, 0]]},
         {"phase_degrees": 180, "receiver_degrees": 0, "signal": [[0.5, 0]]}]
r2 = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1], "acquisitions": rows2})
check("aliased channels reported unidentifiable",
      all(ch["status"] == "unidentifiable" for ch in r2["candidate_channels"]["channels"]))
check("alias class contains +1/-1",
      sorted(r2["identifiability"]["alias_classes"][0]["channels"]) == [-1, 1])
check("unidentifiable signal is null", r2["target"]["signal"] is None)

# 3. States quadrature separates +1/-1 and missing partner is reported (not zero filled).
states = [
    {"phase_degrees": 0, "receiver_degrees": 0, "increment": 0, "quadrature_component": "real",
     "signal": [[truth[1].real + truth[-1].real if -1 in truth else 0, 0]]},
]
# Build properly: amplitudes for +1 and -1
a_plus, a_minus = complex(2, -1), complex(0.5, 0.8)
states = []
for inc, comp, dphase, rshift in [(0, "real", 0, 0), (0, "imaginary", 90, 90),
                                  (1, "real", 0, 0), (1, "imaginary", 90, 90)]:
    # States: same pulse phase within increment, imaginary scan shifts 90.
    # Use distinct physical pulse phases encoded per scan instead:
    pulse = 0 if comp == "real" else 90
    receiver = 0 if comp == "real" else 90
    z = (a_plus * cmath.exp(1j * math.radians(pulse)) +
         a_minus * cmath.exp(-1j * math.radians(pulse)))
    obs = z * cmath.exp(-1j * math.radians(receiver))
    states.append({"phase_degrees": pulse, "receiver_degrees": receiver,
                   "increment": inc, "quadrature_component": comp,
                   "signal": [[obs.real, obs.imag]]})
r3 = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1],
                       "plan": {"quadrature": "states", "increment_count": 2},
                       "acquisitions": states})
check("states separates signs",
      all(ch["status"].startswith("identifiable") for ch in r3["candidate_channels"]["channels"]),
      str(r3["identifiability"]))
plus = next(ch for ch in r3["candidate_channels"]["channels"] if ch["channel"] == 1)
check("states +1 estimate", approx(complex(*plus["signal"][0]), a_plus, 1e-10))
check("states complete quadrature", r3["plan"]["quadrature"]["complete_quadrature"] is True)

# Now drop the imaginary partner of increment 1: must not zero-fill, still rank full?
partial = [a for a in states if not (a["increment"] == 1 and a["quadrature_component"] == "imaginary")]
r3b = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1],
                        "plan": {"quadrature": "states", "increment_count": 2},
                        "acquisitions": partial})
check("missing quadrature partner reported",
      r3b["plan"]["quadrature"]["missing_quadrature_components"] == [{"increment": 1, "missing_component": "imaginary"}])
check("missing partner not zero filled (3 rows used)", r3b["acquisition_count"] == 3)

# 4. Heteroscedastic explicit variances: WLS and known variance gives t-free CIs.
sigma = [0.1, 0.2, 0.15]
noisy_rows = []
for k, s in enumerate(sigma):
    phase = 120 * k
    clean = sum(a * cmath.exp(1j * math.radians(q * phase)) for q, a in truth.items())
    e = complex(rng.gauss(0, s), rng.gauss(0, s))
    noisy_rows.append({"phase_degrees": phase, "receiver_degrees": 0,
                       "signal": [[(clean + e).real, (clean + e).imag]],
                       "variance": 2 * s * s})
r4 = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1],
                       "plan": {"cycle_size": 3}, "acquisitions": noisy_rows})
check("WLS active", r4["variance"]["weighted_least_squares"] is True)
check("Kish effective <= n", r4["variance"]["effective_acquisition"]["maximum"] <= 3 + 1e-12)
check("95% interval present with known variance",
      r4["target"]["confidence_interval_95"][0] is not None)
check("SE positive", r4["target"]["standard_error"][0] > 0)

# 5. Repeated measurements: empirical variance used, status reported.
repeats = []
for k in (0, 1, 2):
    phase = 120 * k
    signals = []
    for _ in range(5):
        clean = sum(a * cmath.exp(1j * math.radians(q * phase)) for q, a in truth.items())
        e = complex(rng.gauss(0, 0.1), rng.gauss(0, 0.1))
        signals.append([[(clean + e).real, (clean + e).imag]])
    repeats.append({"phase_degrees": phase, "receiver_degrees": 0, "signals": signals})
r5 = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1], "acquisitions": repeats})
check("repeat variance status", r5["variance"]["acquisitions"][0]["variance_status"] == "estimated_from_repeats")
check("repeat estimate close", approx(complex(*r5["target"]["signal"][0]), truth[1], 5e-2))
check("total measurements counted", r5["total_measurements"] == 15)

# 6. No variance info at all: OLS, residual df = n-p, unquantified when saturated.
r6 = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1], "acquisitions": rows})
check("OLS residual variance", r6["target"]["status"] == "identifiable_residual_variance")
check("residual df = 1", r6["target"]["residual_degrees_of_freedom"] == 1)
sat_rows = rows[:2]
r6b = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1], "acquisitions": sat_rows})
check("saturated design unquantified", r6b["target"]["status"] == "identifiable_unquantified")
check("saturated CI null", r6b["target"]["confidence_interval_95"][0] is None)

# 7. Blocked estimation matches single-shot on a long time domain.
width = 500
long_rows = []
for k in (0, 1, 2):
    phase = 120 * k
    trace = []
    for t in range(width):
        decay = math.exp(-t / 200.0)
        clean = sum(a * decay * cmath.exp(1j * math.radians(q * phase)) for q, a in truth.items())
        trace.append([clean.real, clean.imag])
    long_rows.append({"phase_degrees": phase, "receiver_degrees": 0, "signal": trace})
r7a = dispatch("/nmr", {"channel": 1, "candidate_channels": [0, 1], "acquisitions": long_rows})
# internal re-run with block size 7
from molecular_materials_foundry.nmr.acquisition_schedule import prepare_general
model = prepare_general({"channel": 1, "candidate_channels": [0, 1], "acquisitions": long_rows})
est_small = estimate(model["basis"], model["rows"], model["candidates"], block_size=7)
est_big = estimate(model["basis"], model["rows"], model["candidates"], block_size=1024)
idx = model["candidates"].index(1)
max_diff = max(abs(a - b) for a, b in zip(est_small["records"][idx]["estimate"],
                                          est_big["records"][idx]["estimate"]))
check("blocked estimates identical", max_diff < 1e-14, f"max diff {max_diff:.2e}")
check("long trace full length", len(r7a["target"]["signal"]) == width)

# 8. Calibration check: coverage near 95%, bias near zero.
cal_rows = [{"phase": 120 * k, "receiver": 0.0, "mean": [0.0 + 0j]} for k in range(3)]
amps = [[complex(1.0, 0.5)], [complex(2.0, -1.0)]]
cal = calibration_check(cal_rows, [0, 1], amps, noise_sigma=0.1, trials=400, seed=42,
                        target_channel=1)
check("calibration all trials identifiable", cal["unidentifiable_trials"] == 0)
check("calibration coverage near nominal", abs(cal["coverage_95"][0] - 0.95) < 0.06,
      f"coverage={cal['coverage_95'][0]:.3f}")
check("calibration bias small", approx(complex(*cal["mean_estimate"][0]), amps[1][0], 2e-2),
      str(cal["mean_estimate"][0]))
check("calibration RMSE near sigma-ish", cal["rmse"][0] < 0.25, f"rmse={cal['rmse'][0]:.3f}")

# 9. States-TPPI convention validation and bad convention rejection.
tppi = []
for inc in range(2):
    for comp in ("real", "imaginary"):
        pulse = 90 * inc + (90 if comp == "imaginary" else 0)
        tppi.append({"phase_degrees": pulse, "receiver_degrees": 0,
                     "increment": inc, "quadrature_component": comp,
                     "signal": [[(a_plus * cmath.exp(1j * math.radians(pulse)) +
                                 a_minus * cmath.exp(-1j * math.radians(pulse))).real,
                                 (a_plus * cmath.exp(1j * math.radians(pulse)) +
                                  a_minus * cmath.exp(-1j * math.radians(pulse))).imag]]})
r9 = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1],
                       "plan": {"quadrature": "states_tppi", "increment_count": 2},
                       "acquisitions": tppi})
check("states-tppi convention valid", r9["plan"]["quadrature"]["phase_convention_valid"] is True)
bad = json.loads(json.dumps(tppi))
bad[2]["phase_degrees"] = 30  # break 90-degree TPPI advance
r9b = dispatch("/nmr", {"channel": 1, "candidate_channels": [-1, 1],
                        "plan": {"quadrature": "states_tppi", "increment_count": 2},
                        "acquisitions": bad})
check("states-tppi violation flagged", r9b["plan"]["quadrature"]["phase_convention_valid"] is False)

# 10. Rejection paths.
for name, payload, code in [
    ("unknown field", {**{"channel": 1, "acquisitions": rows}, "bogus": 1}, "unsupported_field"),
    ("bad variance", {**{"channel": 1, "acquisitions": [{**rows[0], "variance": -1}]}}, "invalid_number"),
    ("signal/signals both", {"channel": 1, "acquisitions": [
        {"phase_degrees": 0, "receiver_degrees": 0, "signal": [[1, 0]],
         "signals": [[[1, 0]], [[1, 0]]]}]}, "invalid_acquisition"),
    ("duplicate pair", {**{"channel": 1, "plan": {"quadrature": "states"},
                           "acquisitions": [states[0], states[0]]}}, "duplicate_quadrature_acquisition"),
    ("phase off grid", {**{"channel": 1, "plan": {"cycle_size": 4},
                           "acquisitions": [{**rows[0], "phase_degrees": 45}, rows[1], rows[2]]}},
     "plan_inconsistent_with_cycle"),
]:
    try:
        dispatch("/nmr", payload)
        check(name, False, "no error raised")
    except DomainError as exc:
        check(name, exc.code == code, f"{exc.code} != {code}")

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
