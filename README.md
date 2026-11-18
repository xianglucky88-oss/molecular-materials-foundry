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
| `/reflections` | Groups reflection indices using explicit reciprocal-space point-group operations and combines independent intensities using known variances. | Operations must be a complete finite group of unimodular integer 3x3 matrices, including identity. Default `friedel: true` merges opposite reflections. Set false to preserve them unless they are already related by an explicitly supplied operation. Does not use space-group translations or infer systematic absences. Negative background-corrected intensities are allowed; variances must be positive. |
| `/speciation` | Computes an ideal mixture of neutral monoprotic acids and their singly negative conjugate bases, including water dissociation and fixed strong-ion charge. | 298.15 K, Kw=1e-14, no nonideal activities, no solid phases, and no general reaction parser. Each acid supplies total concentration in mol/L and pKa. `strong_ion_charge` is fixed positive minus negative spectator-ion concentration, in equivalents/L. pH, mass and charge residuals are returned. |
| `/trajectories` | Restores continuous positions from wrapped coordinates and computes lagged squared displacements. | Fixed orthorhombic box; positions in [0, L). Every frame must supply integer image flags, or every frame must omit them and supply an explicit maximum-step assumption strictly below half the shortest box side. The latter is a conditional inference: actual missed motion cannot be detected from wrapped samples alone. A declared bound violation is rejected. No changing cell, missing-frame interpolation or dynamics simulation. |

Optional trajectory bonds require `max_bond_length`; unwrapped bonds that exceed that declared bound are rejected. Without explicit flags, the first frame defines the atom image origins; the service does not infer a globally unwrapped molecular reference frame. Lag is measured in frame indices and no elapsed-time or diffusion coefficient is inferred. An empty lag window is valid when the trajectory is too short.

A phase-cycle channel is aliased modulo the cycle length; a short schedule cannot uniquely distinguish all integer coherence orders. The service reports this limitation instead of labeling the selected channel as globally unique. Similarly, the restricted stereochemical descriptor is not a full implementation of every CIP sequence rule.

## Responsibility boundaries

Each domain package owns its input contract, scientific calculation and result assembly. No domain package depends on another domain package. The API layer only routes requests; the HTTP module validates JSON framing and maps errors. Common validation contains only primitive type/range checks. Processing is local and stateless and does not require credentials, databases or external services.
