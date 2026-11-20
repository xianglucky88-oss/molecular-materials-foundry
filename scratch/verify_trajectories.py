"""Ad-hoc verification harness (pytest is unavailable in this environment)."""
import json
import sys
import math

sys.path.insert(0, "/workspace/src")
from molecular_materials_foundry.api import dispatch
from molecular_materials_foundry.errors import DomainError

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        failures.append((name, detail))
        print(f"FAIL {name} :: {detail}")


def expect_error(name, payload, code=None):
    try:
        dispatch("/trajectories", payload)
    except DomainError as exc:
        if code is None or exc.code == code:
            print(f"PASS {name} (code={exc.code})")
            return
        failures.append((name, f"wrong code {exc.code}"))
        print(f"FAIL {name} :: wrong code {exc.code}")
        return
    failures.append((name, "no error raised"))
    print(f"FAIL {name} :: no error raised")


# --------------------------------------------------------------------------- #
# 1. Legacy contract is byte-identical in behavior
legacy = json.loads(open("/workspace/examples/trajectories.json").read())
r = dispatch("/trajectories", legacy)
check("legacy model id", r["model"] == "fixed-orthorhombic-cell-v1")
check("legacy unwrap", [list(v) for v in r["continuous_positions"][1]] == [[11, 1, 1]])
check("legacy msd", r["windows"][0]["mean_squared_displacement"] == 4)
check("legacy lag key", r["lag_frames"] == 1)
expect_error("legacy rejects extended field", {**legacy, "groups": []}, "unsupported_field")

# Legacy inferred still works
p = {"box": [10, 10, 10], "atoms": ["a"],
     "frames": [{"positions": [[9, 1, 1]]}, {"positions": [[1, 1, 1]]}], "max_step": 3}
r2 = dispatch("/trajectories", p)
check("legacy inferred", r2["continuous_positions"][1][0] == (11, 1, 1))
expect_error("legacy bound violation", {**p, "max_step": 1}, "step_bound_violation")

# --------------------------------------------------------------------------- #
# 2. Extended explicit images, known motion re-wrapped, two groups
H = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]


def wrap_fractional(cart):
    frac = [c / 10.0 for c in cart]
    img = [int(math.floor(f)) for f in frac]
    wrapped = [f - i for f, i in zip(frac, img)]
    return wrapped, img


# atom a: lab motion (2,0,0) per step; atom b: motion (0,0,0)
truth_a = [[0.1, 0.1, 0.1], [2.1, 0.1, 0.1], [4.1, 0.1, 0.1], [6.1, 0.1, 0.1]]
truth_b = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
frames = []
for fa, fb in zip(truth_a, truth_b):
    wa, ia = wrap_fractional(fa)
    wb, ib = wrap_fractional(fb)
    frames.append({"positions": [wa, wb], "images": [ia, ib]})
ext = {
    "atoms": ["a", "b"],
    "groups": [{"id": "g1", "atoms": ["a"]}, {"id": "g2", "atoms": ["b"]}],
    "times": [0.0, 1.0, 2.0, 3.0],
    "cells": [H, H, H, H],
    "frames": frames,
    "lag_time": 2.0,
    "max_step": 3.0,
}
r = dispatch("/trajectories", ext)
check("ext model id", r["model"] == "triclinic-variable-cell-v1")
check("ext continuous a", all(abs(r["continuous_positions"][i][0][0] - truth_a[i][0]) < 1e-12
                              for i in range(4)))
check("ext window count", len(r["windows"]) == 2)
check("ext msd a window0", abs(r["windows"][0]["squared_displacements"][0] - 16.0) < 1e-12)
g1, g2 = r["group_statistics"]
check("ext group identified g1", g1["identified_samples"] == 2 and g1["candidate_samples"] == 2)
check("ext missing rate zero", g1["missing_rate"] == 0.0 and g2["missing_rate"] == 0.0)
# overlapping windows: ESS must count only one independent window * 1 atom
check("ext ess overlap", g1["effective_sample_size"] == 1, str(g1))
check("ext g2 zero motion", g2["mean_squared_displacement"] == 0.0)
check("ext sampling uniform", r["time_sampling"]["uniform"] is True)
json.dumps(r, allow_nan=False)

# --------------------------------------------------------------------------- #
# 3. Affine cell deformation must not count as particle motion
# particle keeps fractional coordinate 0.1 in x; cell grows 10 -> 11 -> 12.
cells = [
    [[10.0, 0, 0], [0, 10, 0], [0, 0, 10]],
    [[11.0, 0, 0], [0, 10, 0], [0, 0, 10]],
    [[12.0, 0, 0], [0, 10, 0], [0, 0, 10]],
]
frac = [0.1, 0.1, 0.1]
frames = [{"positions": [frac], "images": [[0, 0, 0]]} for _ in range(3)]
base = {"atoms": ["a"], "groups": [{"id": "g", "atoms": ["a"]}],
        "times": [0.0, 1.0, 2.0], "cells": cells, "frames": frames,
        "lag_time": 2.0, "max_step": 5.0}
r_lab = dispatch("/trajectories", {**base, "reference": "laboratory"})
r_aff = dispatch("/trajectories", {**base, "reference": "affine_removed"})
lab_disp_sq = r_lab["windows"][0]["squared_displacements"][0]
aff_disp_sq = r_aff["windows"][0]["squared_displacements"][0]
check("affine lab sees deformation", abs(lab_disp_sq - (1.2 - 1.0) ** 2) < 1e-12,
      str(lab_disp_sq))
check("affine removed zero", aff_disp_sq == 0.0, str(aff_disp_sq))

# --------------------------------------------------------------------------- #
# 3b. Sheared triclinic cell: explicit and bounded inference must agree
H_shear = [[10.0, 0.0, 0.0], [1.0, 8.0, 0.0], [0.0, 0.0, 12.0]]


def wrap_fractional_shear(fractional):
    img = [int(math.floor(f)) for f in fractional]
    return [f - i for f, i in zip(fractional, img)], img


# fractional motion crossing a wrap boundary in a sheared cell; the explicit
# flags follow the minimum-image path (1.6 A step) and a 2 A bound makes it
# unique because every rival wrap exceeds 6 A.
frac_path = [[0.9, 0.95, 0.5], [0.9, 0.15, 0.5], [0.9, 0.35, 0.5]]
shear_frames = []
for fv in frac_path:
    w, img = wrap_fractional_shear(fv)
    shear_frames.append({"positions": [w], "images": [img]})
shear_base = {"atoms": ["a"], "groups": [{"id": "g", "atoms": ["a"]}],
              "times": [0.0, 1.0, 2.0], "cells": [H_shear, H_shear, H_shear],
              "frames": shear_frames, "lag_time": 1.0, "max_step": 4.0}
# explicit flags encode the minimum-image path (image +1 in fractional y)
shear_frames[1]["images"] = [[0, 1, 0]]
shear_frames[2]["images"] = [[0, 1, 0]]
r_expl = dispatch("/trajectories", shear_base)
# same wrapped data without flags: the bound must pick the same unique image
infer_frames = [{"positions": f["positions"]} for f in shear_frames]
r_infer = dispatch("/trajectories", {**shear_base, "frames": infer_frames})
for r_mode, label in ((r_expl, "explicit"), (r_infer, "inferred")):
    disp = [c - a for a, c in zip(r_mode["continuous_positions"][0][0],
                                  r_mode["continuous_positions"][1][0])]
    # H * (0, 0.2, 0) = (0, 1.6, 0)
    expected = [0.0, 1.6, 0.0]
    check(f"shear {label} unique image",
          all(abs(disp[k] - expected[k]) < 1e-9 for k in range(3)), str(disp))
    check(f"shear {label} msd", abs(r_mode["windows"][0]["squared_displacements"][0]
                                    - sum(x * x for x in expected)) < 1e-9)
def close_nested(a, b, tol=1e-9):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close_nested(x, y, tol) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


check("shear modes agree",
      close_nested(r_infer["continuous_positions"], r_expl["continuous_positions"]))
# loosening the bound so a rival wrap qualifies must be reported, not guessed
r_loose = dispatch("/trajectories", {**shear_base, "max_step": 9.0,
                                     "frames": infer_frames})
check("shear loose bound ambiguous",
      not r_loose["windows"][0]["sample_status"][0]["valid"]
      and r_loose["windows"][0]["sample_status"][0]["reason"] == "ambiguous_image")
# explicit flags contradicting the bound remain a hard rejection
bad_flags = json.loads(json.dumps(shear_base))
bad_flags["frames"][1]["images"] = [[0, 0, 0]]
bad_flags["frames"][2]["images"] = [[0, 0, 0]]
expect_error("shear explicit bound reject", bad_flags, "step_bound_violation")

# --------------------------------------------------------------------------- #
# 4. Non-uniform times: lag matching picks nearest, unmatched recorded
base4 = dict(ext, times=[0.0, 1.0, 1.5, 3.0], lag_time=2.0, lag_tolerance=0.6)
r = dispatch("/trajectories", base4)
check("nonuniform flag", r["time_sampling"]["uniform"] is False)
# t=0 -> t=1.5 (err .5); t=1 -> t=3 (exact); t=1.5 -> target 3.5, t=3 err .5 -> frame3
check("nonuniform pairs", len(r["windows"]) == 3
      and r["lag_matching"]["unmatched_frame_starts"] == [3], str(r["lag_matching"]))
check("nonuniform elapsed", [w["elapsed_time"] for w in r["windows"]] == [1.5, 2.0, 1.5])
# lag beyond the recording yields zero windows but is itself a valid request
r_far = dispatch("/trajectories", dict(ext, times=[0, 1, 1.5, 3], lag_time=5.0))
check("unreachable lag empty", r_far["windows"] == []
      and r_far["lag_matching"]["unmatched_frame_starts"] == [0, 1, 2, 3])

# --------------------------------------------------------------------------- #
# 5. Missing frame contaminates every window spanning the gap
miss = json.loads(json.dumps(ext))
miss["frames"][2] = {"positions": None}
miss["lag_time"] = 1.0
r = dispatch("/trajectories", miss)
# pairs t0->t1, t1->t2, t2->t3; middle pair endpoints missing; last spans gap
statuses = [[s["valid"] for s in w["sample_status"]] for w in r["windows"]]
check("missing endpoint", statuses[1] == [False, False])
check("missing gap propagation", statuses[2] == [False, False])
check("missing first window fine", statuses[0] == [True, True])
g1 = r["group_statistics"][0]
check("missing rate", abs(g1["missing_rate"] - 2 / 3) < 1e-12, str(g1["missing_rate"]))
check("missing ess", g1["effective_sample_size"] == 1)
# conservative envelope present and finite
check("missing envelope", g1["msd_conservative_envelope"] is not None
      and g1["msd_conservative_envelope"][1] > 0)
check("anomaly listed", g1["anomalous_intervals_count"] == 2
      and g1["anomalous_intervals"][0]["reason"] == "missing_frame")

# --------------------------------------------------------------------------- #
# 6. Multi-candidate images: ambiguity propagated, not guessed
# atom moves from frac .95 to .05 (true +.10 via image -1); bound allows both
amb = {
    "atoms": ["a"], "groups": [{"id": "g", "atoms": ["a"]}],
    "times": [0.0, 1.0, 2.0],
    "cells": [H, H, H],
    "frames": [{"positions": [[0.95, 0.1, 0.1]]},
               {"positions": [[0.05, 0.1, 0.1]]},
               {"positions": [[0.10, 0.1, 0.1]]}],
    "lag_time": 1.0, "max_step": 6.0,
}
r = dispatch("/trajectories", amb)
# step .95->.05: shifts give cartesian lengths 9 (m=0) and 1 (m=-1): 9 > 6 -> unique length-1.
amb["max_step"] = 9.0
r = dispatch("/trajectories", amb)
# now step0 has lengths 9 and 1, both <= 9 -> ambiguous; step2 delta .05 gives 9.5 vs .5 -> unique
check("ambiguous window0", r["windows"][0]["sample_status"][0]["valid"] is False
      and r["windows"][0]["sample_status"][0]["reason"] == "ambiguous_image")
check("unique window1", r["windows"][1]["sample_status"][0]["valid"] is True)
g = r["group_statistics"][0]
check("ambiguity missing rate", g["missing_rate"] == 0.5)
check("ambiguity keeps interval wide", g["msd_conservative_envelope"] is not None)

# no candidate at all -> hard rejection
amb2 = dict(amb, max_step=0.5)
expect_error("bound violation triclinic", amb2, "step_bound_violation")

# --------------------------------------------------------------------------- #
# 7. Explicit bond closure conflict propagates, legacy bond still hard-rejects
bond_ext = {
    "atoms": ["a", "b"],
    "groups": [{"id": "g", "atoms": ["a", "b"]}],
    "times": [0.0, 1.0], "cells": [H, H],
    "frames": [
        {"positions": [[0.95, 0, 0], [0.05, 0, 0]], "images": [[0, 0, 0], [1, 0, 0]]},
        {"positions": [[0.95, 0, 0], [0.05, 0, 0]], "images": [[0, 0, 0], [0, 0, 0]]},
    ],
    "lag_time": 1.0, "max_step": 11.0,
    "bonds": [["a", "b"]], "max_bond_length": 1.1,
}
r = dispatch("/trajectories", bond_ext)
check("bond conflict invalid", r["windows"][0]["sample_status"][0]["valid"] is False
      and r["windows"][0]["sample_status"][1]["valid"] is False)
check("bond conflict reason", r["windows"][0]["sample_status"][0]["reason"] == "bond_conflict")
legacy_bond = {"box": [10, 10, 10], "atoms": ["a", "b"],
               "frames": [{"positions": [[9.5, 0, 0], [0.5, 0, 0]],
                           "images": [[0, 0, 0], [0, 0, 0]]}],
               "bonds": [["a", "b"]], "max_bond_length": 1.1}
expect_error("legacy bond hard reject", legacy_bond, "bond_image_conflict")

# --------------------------------------------------------------------------- #
# 8. Inferred mode with bond resolves the correct image jointly
bond_inf = {
    "atoms": ["a", "b"],
    "groups": [{"id": "g", "atoms": ["a", "b"]}],
    "times": [0.0, 1.0], "cells": [H, H],
    "frames": [
        {"positions": [[0.95, 0, 0], [0.05, 0, 0]]},
        {"positions": [[0.05, 0, 0], [0.15, 0, 0]]},
    ],
    "lag_time": 1.0, "max_step": 9.5,
    "bonds": [["a", "b"]], "max_bond_length": 1.1,
}
r = dispatch("/trajectories", bond_inf)
# bond requires a shift of +1 for atom a (distance 1.0); atom b unique +0? b .05->.15 d=.1
check("bond resolves joint", all(s["valid"] for s in r["windows"][0]["sample_status"]),
      str(r["windows"][0]["sample_status"]))
# without bond, atom a would be ambiguous -> compare
bond_inf2 = {k: v for k, v in bond_inf.items() if k not in ("bonds", "max_bond_length")}
r2 = dispatch("/trajectories", bond_inf2)
check("no bond ambiguous", not r2["windows"][0]["sample_status"][0]["valid"])

# --------------------------------------------------------------------------- #
# 9. Validation errors
expect_error("group partition", {**ext, "groups": [{"id": "g", "atoms": ["a"]}]},
             "group_partition")
expect_error("times count", {**ext, "times": [0, 1]}, "invalid_array")
expect_error("nonmono times", {**ext, "times": [0, 1, 1, 3]}, "invalid_time")
expect_error("degenerate cell", {**ext, "cells": [[[0, 0, 0], [0, 10, 0], [0, 0, 10]]] * 4},
             "degenerate_cell")
expect_error("wrapped range", {**ext,
                               "frames": [dict(f, positions=[[1.2, 0, 0], f["positions"][1]])
                                          for f in ext["frames"]]}, "unwrapped_input")
expect_error("mixed images", {**ext,
                              "frames": [dict(f) for f in ext["frames"][:1]]
                              + [{"positions": ext["frames"][1]["positions"]}]
                              + ext["frames"][2:]}, "mixed_image_mode")
expect_error("bad reference", {**ext, "reference": "guess"}, "invalid_reference")

# exact tie between two end frames for the requested lag
tie = dict(ext, times=[0.0, 0.5, 1.0, 1.5], lag_time=0.75, lag_tolerance=0.3)
# start 0: target 0.75 -> frames at 0.5 and 1.0 are equidistant (0.25 each)
expect_error("ambiguous lag tie", tie, "ambiguous_lag_match")

# empty matched set: group stats report missing_rate None and null intervals
empty = dict(ext, lag_time=20.0)
r = dispatch("/trajectories", empty)
g = r["group_statistics"][0]
check("empty lag stats", g["identified_samples"] == 0 and g["missing_rate"] is None
      and g["msd_conservative_envelope"] is None and g["distribution"]["count"] == 0)

print()
if failures:
    print(f"{len(failures)} FAILURES")
    for name, detail in failures:
        print(" -", name, detail)
    sys.exit(1)
print("ALL CHECKS PASSED")
