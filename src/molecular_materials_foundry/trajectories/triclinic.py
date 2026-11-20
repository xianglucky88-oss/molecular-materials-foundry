"""Per-frame triclinic cells, fractional unwrapping and continuity resolution.

Extended trajectory analysis.  The legacy fixed orthorhombic contract lives in
``cell_images`` / ``molecular_continuity`` and is intentionally untouched.

Coordinates in this module are *fractional* wrapped positions ``s`` in [0, 1);
cartesian positions are ``x = H s + H n`` where ``H`` is the 3x3 lattice-vector
matrix (rows are the ``a, b, c`` vectors) and ``n`` an integer image vector.
"""
import itertools
import math
from ..validation import fields, sequence, number, integer, label
from ..errors import DomainError

MAX_IMAGE_FLAG = 10**6
WORK_ENTRIES = 100_000
MAX_FRAMES = 1000
MAX_ATOMS = 1000
MIN_DETERMINANT = 1e-12
MAX_CONDITION = 1e12
# Enumeration guards so an over-loose motion bound cannot explode the search.
CANDIDATE_LIST_CAP = 64
ENUMERATION_CAP = 200_000
JOINT_SEARCH_CAP = 20_000
ANOMALY_LIST_CAP = 100


# --------------------------------------------------------------------------- #
# Small matrix helpers (standard library only)
# --------------------------------------------------------------------------- #
def matvec(matrix, vector):
    return tuple(math.fsum(row[j] * vector[j] for j in range(3)) for row in matrix)


def determinant(matrix):
    a, b, c = matrix
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def inverse(matrix):
    """Invert a 3x3 matrix with partial pivoting; reject degenerate cells."""
    work = [list(row) + [1.0 if i == j else 0.0 for j in range(3)]
            for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < 1e-300:
            raise DomainError("degenerate_cell", "Cell matrix is singular")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
        pivot_value = work[col][col]
        for j in range(6):
            work[col][j] /= pivot_value
        for r in range(3):
            if r == col:
                continue
            factor = work[r][col]
            if factor:
                for j in range(6):
                    work[r][j] -= factor * work[col][j]
    return tuple(tuple(row[3:]) for row in work)


def matrix_norm_inf(matrix):
    return max(sum(abs(value) for value in row) for row in matrix)


def euclidean(vector):
    return math.sqrt(math.fsum(value * value for value in vector))


def _cell_matrix(value):
    rows = sequence(value, "cell", minimum=3, maximum=3)
    matrix = []
    for row in rows:
        entries = sequence(row, "cell row", minimum=3, maximum=3)
        matrix.append(tuple(number(x, "cell entry", minimum=-1e9, maximum=1e9) for x in entries))
    determinant_value = determinant(matrix)
    if determinant_value <= MIN_DETERMINANT:
        raise DomainError("degenerate_cell", "Cell volume must be positive and non-degenerate")
    inverse_matrix = inverse(matrix)
    condition = matrix_norm_inf(matrix) * matrix_norm_inf(inverse_matrix)
    if condition > MAX_CONDITION:
        raise DomainError("degenerate_cell", "Cell matrix is too ill-conditioned")
    return tuple(matrix), inverse_matrix


# --------------------------------------------------------------------------- #
# Request parsing
# --------------------------------------------------------------------------- #
def prepare_extended(payload):
    fields(payload, ["atoms", "groups", "times", "cells", "frames", "lag_time", "max_step"],
           ["lag_tolerance", "reference", "bonds", "max_bond_length"])
    atoms = tuple(label(x, "atom id") for x in sequence(payload["atoms"], "atoms",
                                                        minimum=1, maximum=MAX_ATOMS))
    if len(set(atoms)) != len(atoms):
        raise DomainError("duplicate_atom", "Atom identifiers must be distinct")
    groups = _prepare_groups(payload["groups"], atoms)

    frame_count = len(sequence(payload["frames"], "frames", minimum=1, maximum=MAX_FRAMES))
    times = tuple(number(x, "frame time", minimum=-1e15, maximum=1e15)
                  for x in sequence(payload["times"], "times",
                                    minimum=frame_count, maximum=frame_count))
    if any(times[i] >= times[i + 1] for i in range(frame_count - 1)):
        raise DomainError("invalid_time", "Frame times must be strictly increasing")
    raw_cells = sequence(payload["cells"], "cells", minimum=frame_count, maximum=frame_count)
    cells, cell_inverses = [], []
    for raw in raw_cells:
        matrix, inverse_matrix = _cell_matrix(raw)
        cells.append(matrix)
        cell_inverses.append(inverse_matrix)

    frames = []
    image_modes = set()
    for frame in sequence(payload["frames"], "frames", minimum=1, maximum=MAX_FRAMES):
        fields(frame, ["positions"], ["images"])
        if frame["positions"] is None:
            if "images" in frame:
                raise DomainError("missing_frame", "A missing frame cannot carry image flags")
            frames.append({"observed": False, "positions": None, "images": None})
            continue
        positions = []
        for row in sequence(frame["positions"], "positions",
                            minimum=len(atoms), maximum=len(atoms)):
            xyz = tuple(number(x, "fractional position", minimum=0.0)
                        for x in sequence(row, "position", minimum=3, maximum=3))
            if any(value >= 1.0 for value in xyz):
                raise DomainError("unwrapped_input",
                                  "Triclinic positions must be fractional and wrapped into [0, 1)")
            positions.append(xyz)
        images = None
        if "images" in frame:
            image_modes.add(True)
            images = [tuple(integer(x, "image", minimum=-MAX_IMAGE_FLAG, maximum=MAX_IMAGE_FLAG)
                            for x in sequence(row, "image", minimum=3, maximum=3))
                      for row in sequence(frame["images"], "images",
                                          minimum=len(atoms), maximum=len(atoms))]
        else:
            image_modes.add(False)
        frames.append({"observed": True, "positions": tuple(positions), "images": images})
    if len(image_modes) > 1:
        raise DomainError("mixed_image_mode",
                          "Every observed frame must provide images, or every frame must omit them")
    if frame_count * len(atoms) > WORK_ENTRIES:
        raise DomainError("work_limit", "Use at most 100000 frame-atom entries")

    max_step = number(payload["max_step"], "max_step", minimum=0.0)
    lag_time = number(payload["lag_time"], "lag_time", positive=True, maximum=1e15)
    tolerance = payload.get("lag_tolerance", 1e-9 * max(1.0, abs(lag_time)))
    lag_tolerance = number(tolerance, "lag_tolerance", minimum=0.0)
    reference = payload.get("reference", "laboratory")
    if reference not in ("laboratory", "affine_removed"):
        raise DomainError("invalid_reference",
                          "reference must be 'laboratory' or 'affine_removed'")
    bonds, bond_limit = _prepare_bonds(payload, atoms)
    return {
        "atoms": atoms, "groups": groups, "times": times,
        "cells": tuple(cells), "cell_inverses": tuple(cell_inverses),
        "frames": frames, "image_mode": "explicit" if image_modes == {True}
        else "bounded_step_assumption",
        "max_step": max_step, "lag_time": lag_time, "lag_tolerance": lag_tolerance,
        "reference": reference, "bonds": bonds, "bond_limit": bond_limit,
    }


def _prepare_groups(raw_groups, atoms):
    groups = []
    known = set(atoms)
    seen_ids = set()
    assigned = set()
    for raw in sequence(raw_groups, "groups", minimum=1, maximum=100):
        fields(raw, ["id", "atoms"])
        group_id = label(raw["id"], "group id")
        if group_id in seen_ids:
            raise DomainError("duplicate_group", "Group identifiers must be distinct")
        seen_ids.add(group_id)
        members = tuple(label(x, "group atom")
                        for x in sequence(raw["atoms"], "group atoms", minimum=1))
        if any(name not in known for name in members):
            raise DomainError("invalid_group", "Groups may only reference known atoms")
        if len(set(members)) != len(members) or any(name in assigned for name in members):
            raise DomainError("group_partition", "Groups must partition the atoms without overlap")
        assigned.update(members)
        groups.append((group_id, members))
    if assigned != known:
        raise DomainError("group_partition", "Every atom must belong to exactly one group")
    return tuple(groups)


def _prepare_bonds(payload, atoms):
    bonds = sequence(payload.get("bonds", []), "bonds", maximum=10000)
    if bool(bonds) != ("max_bond_length" in payload):
        raise DomainError("bond_contract",
                          "Nonempty bonds and max_bond_length must be supplied together")
    if not bonds:
        return (), None
    limit = number(payload["max_bond_length"], "max_bond_length", positive=True, maximum=1e12)
    indices = {name: i for i, name in enumerate(atoms)}
    seen = set()
    parsed = []
    for raw in bonds:
        names = [label(x, "bond atom") for x in sequence(raw, "bond", minimum=2, maximum=2)]
        if any(name not in indices for name in names) or names[0] == names[1]:
            raise DomainError("invalid_bond", "Bond references must name distinct known atoms")
        key = tuple(sorted(names))
        if key in seen:
            raise DomainError("duplicate_bond", "Each bond must be specified once")
        seen.add(key)
        parsed.append((indices[key[0]], indices[key[1]]))
    return tuple(parsed), limit


def bond_components(bonds, atom_count):
    adjacency = [set() for _ in range(atom_count)]
    for a, b in bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    components = []
    visited = set()
    for root in range(atom_count):
        if root in visited:
            continue
        order = []
        stack = [root]
        while stack:
            atom = stack.pop()
            if atom in visited:
                continue
            visited.add(atom)
            order.append(atom)
            stack.extend(sorted(adjacency[atom] - visited))
        components.append(order)
    return adjacency, components


# --------------------------------------------------------------------------- #
# Candidate image enumeration for a single inferred consecutive step
# --------------------------------------------------------------------------- #
def candidate_shifts(cell, cell_inverse, delta_fractional, max_step):
    """All integer image shifts ``m`` with ``|H (d - m)| <= max_step``.

    Returns ``(shifts, truncated)``.  Shells around the nearest-integer shift are
    searched outward; a norm bound proves when further shells cannot contribute.
    """
    nearest = tuple(math.floor(value + 0.5) for value in delta_fractional)
    inverse_norm = math.sqrt(3.0) * matrix_norm_inf(cell_inverse)
    scale = max(1.0, max_step)
    bound = max_step + 1e-12 * max_step + 32.0 * math.ulp(scale)
    found = []
    enumerated = 0
    radius = 0
    while True:
        # Every point of the full L_inf shell is reached at its L_inf radius;
        # the norm-inverse bound proves when outer shells cannot contribute.
        if radius and radius / inverse_norm > bound:
            break
        axes = [range(nearest[a] - radius, nearest[a] + radius + 1) for a in range(3)]
        for shift in itertools.product(*axes):
            if radius and max(abs(shift[a] - nearest[a]) for a in range(3)) != radius:
                continue
            enumerated += 1
            vector = tuple(delta_fractional[a] - shift[a] for a in range(3))
            if euclidean(matvec(cell, vector)) <= bound:
                found.append(shift)
            if enumerated > ENUMERATION_CAP:
                return found, True
        radius += 1
        if radius > MAX_IMAGE_FLAG:
            return found, True
    return found, False


def _resolve_component(members, adjacency, candidates, positions, cell, bond_limit):
    """Jointly pick image shifts for one bonded component.

    Returns ``(status, assignments)`` where status is ``identified``,
    ``ambiguous_image`` or ``bond_conflict``.
    """
    if any(candidates[atom] is None or len(candidates[atom]) > CANDIDATE_LIST_CAP
           for atom in members):
        return "ambiguous_image", {}
    order = [members[0]]
    for atom in members[1:]:
        order.append(atom)
    assignments = {}
    solutions = []
    explored = 0

    def backtrack(index):
        nonlocal explored
        if len(solutions) > 1:
            return
        if index == len(order):
            solutions.append(dict(assignments))
            return
        atom = order[index]
        for shift in candidates[atom]:
            explored += 1
            if explored > JOINT_SEARCH_CAP:
                solutions.append({})
                solutions.append({})
                return
            feasible = True
            for neighbour in adjacency[atom]:
                if neighbour not in assignments:
                    continue
                here = matvec(cell, tuple(positions[atom][k] + shift[k] for k in range(3)))
                there = matvec(cell, tuple(positions[neighbour][k]
                                           + assignments[neighbour][k] for k in range(3)))
                if euclidean(tuple(x - y for x, y in zip(here, there))) > bond_limit:
                    feasible = False
                    break
            if feasible:
                assignments[atom] = shift
                backtrack(index + 1)
                del assignments[atom]

    backtrack(0)
    if not solutions:
        return "bond_conflict", {}
    if len(solutions) > 1 or any(not solution for solution in solutions):
        return "ambiguous_image", {}
    return "identified", solutions[0]


# --------------------------------------------------------------------------- #
# Full trajectory reconstruction
# --------------------------------------------------------------------------- #
def reconstruct_extended(config):
    atoms = config["atoms"]
    atom_count = len(atoms)
    frames = config["frames"]
    cells = config["cells"]
    frame_count = len(frames)
    adjacency, components = bond_components(config["bonds"], atom_count)

    continuous = [[None] * atom_count for _ in range(frame_count)]
    segments = [[None] * atom_count for _ in range(frame_count)]
    # Reason recorded per atom on the first frame of every broken segment.
    break_reasons = {}
    conflict_frames = set()
    next_segment = 1

    if config["image_mode"] == "explicit":
        cartesian = [[None] * atom_count for _ in range(frame_count)]
        segment = None
        for f, frame in enumerate(frames):
            if not frame["observed"]:
                segment = None
                continue
            if segment is None:
                segment = next_segment
                next_segment += 1
            for a in range(atom_count):
                fractional = tuple(frame["positions"][a][k] + frame["images"][a][k]
                                   for k in range(3))
                cartesian[f][a] = matvec(cells[f], fractional)
                continuous[f][a] = cartesian[f][a]
                segments[f][a] = segment
        # Declared motion bound is falsified if a consecutive step exceeds it.
        for f in range(1, frame_count):
            if not frames[f]["observed"] or not frames[f - 1]["observed"]:
                continue
            for a in range(atom_count):
                step = tuple(x - y for x, y in zip(cartesian[f][a], cartesian[f - 1][a]))
                if euclidean(step) > config["max_step"] + 1e-12 * config["max_step"] \
                        + 32.0 * math.ulp(max(1.0, config["max_step"])):
                    raise DomainError("step_bound_violation",
                                      "Explicit images violate the declared displacement bound")
        # Explicit image flags can still contradict declared bond closures.
        if config["bonds"]:
            for f, frame in enumerate(frames):
                if not frame["observed"]:
                    continue
                for a, b in config["bonds"]:
                    distance = euclidean(tuple(x - y for x, y
                                               in zip(cartesian[f][a], cartesian[f][b])))
                    if distance > config["bond_limit"]:
                        conflict_frames.add((f, a))
                        conflict_frames.add((f, b))
        return {"continuous": continuous, "segments": segments,
                "break_reasons": break_reasons, "conflict_frames": conflict_frames,
                "adjacency": adjacency, "components": components}

    # Bounded step assumption: incremental minimum-image reconstruction.
    max_step = config["max_step"]
    for a in range(atom_count):
        if frames[0]["observed"]:
            continuous[0][a] = tuple(frames[0]["positions"][a])
            segments[0][a] = 0
    for f in range(1, frame_count):
        frame, previous = frames[f], frames[f - 1]
        if not frame["observed"]:
            continue
        if not previous["observed"]:
            for a in range(atom_count):
                continuous[f][a] = tuple(frame["positions"][a])
                segments[f][a] = next_segment
                break_reasons[(f, a)] = "missing_frame"
            next_segment += 1
            continue
        outcomes = {}
        for component in components:
            if len(component) == 1 and not adjacency[component[0]]:
                atom = component[0]
                delta = tuple(frame["positions"][atom][k] - previous["positions"][atom][k]
                              for k in range(3))
                shifts, truncated = candidate_shifts(cells[f], config["cell_inverses"][f],
                                                      delta, max_step)
                if not shifts:
                    raise DomainError("step_bound_violation",
                                      "No periodic image satisfies the declared displacement bound")
                if truncated or len(shifts) > 1:
                    outcomes[atom] = ("ambiguous_image", None)
                else:
                    outcomes[atom] = ("identified", shifts[0])
                continue
            candidates = {}
            for atom in component:
                delta = tuple(frame["positions"][atom][k] - previous["positions"][atom][k]
                              for k in range(3))
                shifts, truncated = candidate_shifts(cells[f], config["cell_inverses"][f],
                                                      delta, max_step)
                if not shifts:
                    raise DomainError("step_bound_violation",
                                      "No periodic image satisfies the declared displacement bound")
                candidates[atom] = None if truncated else shifts
            status, joint = _resolve_component(component, adjacency, candidates,
                                               frame["positions"], cells[f],
                                               config["bond_limit"])
            for atom in component:
                outcomes[atom] = (status, joint.get(atom))
        for a in range(atom_count):
            status, shift = outcomes[a]
            if status == "identified":
                delta = tuple(frame["positions"][a][k] - previous["positions"][a][k]
                              for k in range(3))
                continuous[f][a] = tuple(continuous[f - 1][a][k] + delta[k] - shift[k]
                                         for k in range(3))
                segments[f][a] = segments[f - 1][a]
            else:
                continuous[f][a] = tuple(frame["positions"][a])
                segments[f][a] = next_segment
                next_segment += 1
                break_reasons[(f, a)] = status

    # Convert continuous fractional coordinates to cartesian laboratory frame.
    cartesian = [[None] * atom_count for _ in range(frame_count)]
    for f, frame in enumerate(frames):
        if not frame["observed"]:
            continue
        for a in range(atom_count):
            cartesian[f][a] = matvec(cells[f], continuous[f][a])
    return {"continuous_fractional": continuous, "continuous": cartesian,
            "segments": segments, "break_reasons": break_reasons,
            "conflict_frames": conflict_frames, "adjacency": adjacency,
            "components": components}
