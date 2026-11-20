# Molecular Materials Foundry

Independent molecular and materials data-processing components, exposed as a Python library and local JSON service. Runtime: Python 3.11+ standard library. Tests: pytest 8.x or 9.x. Wheel build backend: setuptools 68+.

## Run

```bash
python -m pip install .
python -m molecular_materials_foundry --port 9841
curl -H 'Content-Type: application/json' --data-binary @examples/isotopes.json http://127.0.0.1:9841/isotopes
PYTHONPATH=src python -m pytest
```

`GET /health` lists six routes. POST requests use `application/json`, with at most 2 MiB per body. Responses are `{"result": ...}`; invalid scientific requests return HTTP 422 with a structured error. Duplicate JSON keys and nonfinite values are rejected. The default listener is loopback; external deployment and authentication are not configured.

As a library, use `molecular_materials_foundry.api.dispatch(path, payload)` or individual component `process` functions. `examples/` contains a complete request for each route.

## Processing contracts

| Endpoint | Meaning | Supported scope |
|---|---|---|
| `/nmr` | Selects a coherence-channel residue from complex time-domain acquisitions and their pulse/receiver phases. Each complex sample is `[real, imaginary]`. | One complete uniformly spaced single-axis cycle, 2 to 64 acquisitions. Observed signal = physical signal * exp(-i * receiver phase); a channel q contributes exp(i * q * pulse phase). Channel q is identified modulo the acquisition count, reported as `alias_modulus`. Multiple indirect dimensions and incomplete schedules are rejected. |
| `/isotopes` | Enumerates complete isotope composition probabilities and neutral mass or ion mass/charge. | H/C/N/O/P default populations or explicit `[mass_u, probability]` arrays. The embedded numbers are fixed example natural-abundance conventions, not a universal abundance certificate. Exact composition enumeration up to 100000 states; no probability cutoff, instrument broadening or adduct inference. Ion mass subtracts charge times electron mass. Custom populations must sum to one. Numerically unrepresentable nonzero probabilities cause an error. |
| `/stereo` | Reports a tetrahedral descriptor when immediate ligand priorities and geometry resolve it. | One connected acyclic single-bond graph; all hydrogens explicit. Immediate priority uses atomic number and, for same-element explicitly labeled isotopes, mass number. Four distinct immediate priorities are required. Deeper ligand comparisons, rings and multiple bonds are outside this scope. Coordinates must describe the tetrahedral atom arrangement with the center strictly inside the ligand tetrahedron, not a 2D drawing. Unknown priority and degenerate geometry produce explicit statuses, not guessed descriptors. |
| `/reflections` | Groups reflection indices using explicit reciprocal-space operations and combines independent intensities using known variances. Two request shapes are supported. | **Single set:** explicit complete finite group of unimodular integer 3x3 matrices, including identity. Default `friedel: true` merges opposite reflections. Set false to preserve them unless already related by a supplied operation. No translations or inferred systematic absences. **Multi-batch handoff (`batches`):** each batch declares an optional `basis_change` (unimodular det +1 integer index map), an optional affine `space_group` (rotation plus translation in twelfths of a lattice vector), and an `anomalous` convention (`preserve` or `merge`). Relative scales come only from overlapping reflections; batches without a supported overlap stay in separate scale groups and are never force-merged. Negative background-corrected intensities are kept; non-positive variances exclude the observation. Systematic absences, unmeasured reflections, scale outliers and between-batch tension are reported, not silently resolved. |
| `/speciation` | Computes an ideal mixture of neutral monoprotic acids and their singly negative conjugate bases, including water dissociation and fixed strong-ion charge. | 298.15 K, Kw=1e-14, no nonideal activities, no solid phases, and no general reaction parser. Each acid supplies total concentration in mol/L and pKa. `strong_ion_charge` is fixed positive minus negative spectator-ion concentration, in equivalents/L. pH, mass and charge residuals are returned. |
| `/trajectories` | Restores continuous positions from wrapped coordinates and computes lagged squared displacements. | Fixed orthorhombic box; positions in [0, L). Every frame must supply integer image flags, or every frame must omit them and supply an explicit maximum-step assumption strictly below half the shortest box side. The latter is a conditional inference: actual missed motion cannot be detected from wrapped samples alone. A declared bound violation is rejected. No changing cell, missing-frame interpolation or dynamics simulation. |

Optional trajectory bonds require `max_bond_length`; unwrapped bonds that exceed that declared bound are rejected. Without explicit flags, the first frame defines the atom image origins; the service does not infer a globally unwrapped molecular reference frame. Lag is measured in frame indices and no elapsed-time or diffusion coefficient is inferred. An empty lag window is valid when the trajectory is too short.

A phase-cycle channel is aliased modulo the cycle length; a short schedule cannot uniquely distinguish all integer coherence orders. The service reports this limitation instead of labeling the selected channel as globally unique. Similarly, the restricted stereochemical descriptor is not a full implementation of every CIP sequence rule.

### Multi-batch reflection handoff

A `batches` request is for combining several measurement runs that may use different index frames. Each batch declares:

- `basis_change` (optional): a unimodular, handedness-preserving (determinant +1) integer 3x3 matrix mapping batch indices into the common reference frame, `h_ref = h_batch P`. A handedness-reversing map is rejected. All batch space groups must become identical after their basis changes.
- `space_group` (optional): a closed affine group of `{rotation, translation}` operations. Translations are multiples of 1/12 of a lattice vector (this covers denominators 1, 2, 3, 4, 6 and 12). When absent, only the identity applies.
- `anomalous`: `preserve` keeps Friedel opposites distinguishable; `merge` means the batch already averaged them. With the request-level `friedel: false`, a merged observation cannot populate either preserved opposite and is listed under `excluded_observations`; it is also excluded from the overlap scale fit.
- its own `observations`, each `{hkl, intensity, variance}`. Negative background-corrected intensities are valid. A non-positive variance makes the observation statistically unusable; it is retained in the document with `included: false` and never enters a weighted mean.

Relative scales are estimated pair by pair from reflections the two batches measure in common. A pair needs at least two shared, signal-bearing reflections and a relative standard error no worse than 50%. The weighted least-squares ratio is initialized at the median per-reflection ratio so a single gross outlier cannot mask itself; classes beyond three standardised residuals are reported as scale outliers and left out of the fit while still shown in the result. Supported pairs define an overlap graph: connected batches receive one anchor-relative scale each, and closed cycles return a consistency chi-squared test. Batches in different graph components have no supported relative scale. They are never merged across components: a reflection seen by two such components is returned with `status: "scale_groups_not_comparable"` and one intensity per scale group instead of an invented combined value.

Every merged reflection reports its contributing batches, the scaled mean and variance of each, per-observation weights, standardised residuals, leave-one-out shifts and outlier flags. Outliers are flagged but retained — deleting them would overstate precision. Legal negative intensities survive with `status: "negative_intensity"`. When `friedel: false`, surviving opposite pairs carry a `friedel_contrast` with the delta and its variance.

Systematic absences are derived from the affine group: an operation that maps a reflection onto itself (or onto its Friedel opposite when opposites are merged) with a non-integer translation phase makes that reflection extinct. Absences are reported separately from `unmeasured_reflections` (allowed indices supplied through `expected_reflections` that no batch measured). A measured intensity at an extinct orbit is surfaced as an `extinction_conflict` warning rather than being hidden. `examples/reflections_batches.json` exercises a scaled batch on a changed basis, a non-comparable batch, absences and unmeasured indices together.

## Responsibility boundaries

Each domain package owns its input contract, scientific calculation and result assembly. No domain package depends on another domain package. The API layer only routes requests; the HTTP module validates JSON framing and maps errors. Common validation contains only primitive type/range checks. Processing is local and stateless and does not require credentials, databases or external services.
