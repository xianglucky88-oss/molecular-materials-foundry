"""Structural validation of States / States-TPPI quadrature declarations.

The regression itself uses explicit pulse and receiver phases, so the
quadrature declaration's job is to verify the documented pair structure,
report missing real/imaginary partners and explain which sign-discriminating
information the schedule actually carries.  Missing partners are never
replaced by zeros: the surviving acquisition remains a real observation and
the missing component is reported.
"""
from ..errors import DomainError


def _close(left, right, tolerance):
    return abs((left - right + 180.0) % 360.0 - 180.0) <= tolerance


def validate_quadrature(rows, mode, declared_count, tolerance):
    if mode == "none":
        return {"mode": "none", "declared": False}
    for row in rows:
        if row["increment"] is None or row["component"] is None:
            raise DomainError("invalid_quadrature",
                              "States/States-TPPI acquisitions require increment and quadrature_component")
    keys = [(row["increment"], row["component"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise DomainError("duplicate_quadrature_acquisition",
                          "Each (increment, quadrature_component) pair may appear only once")

    by_increment = {}
    for row in rows:
        by_increment.setdefault(row["increment"], {})[row["component"]] = row
    increments = sorted(by_increment)
    violations = []
    if declared_count is not None:
        if any(increment >= declared_count for increment in increments):
            raise DomainError("increment_outside_plan",
                              "An acquisition increment exceeds the declared increment_count")
    reference = {component: [row for row in rows if row["component"] == component]
                 for component in ("real", "imaginary")}

    if mode == "states":
        for component, entries in reference.items():
            if entries:
                phase0 = entries[0]["phase"]
                receiver0 = entries[0]["receiver"]
                for row in entries[1:]:
                    if not _close(row["phase"], phase0, tolerance):
                        violations.append(f"{component} pulse phase is not constant across increments")
                        break
                for row in entries[1:]:
                    if not _close(row["receiver"], receiver0, tolerance):
                        violations.append(f"{component} receiver phase is not constant across increments")
                        break
    else:  # states_tppi
        real_rows = sorted(reference["real"], key=lambda row: row["increment"])
        for earlier, later in zip(real_rows, real_rows[1:]):
            if later["increment"] - earlier["increment"] != 1:
                continue
            if not _close(later["phase"] - earlier["phase"], 90.0, tolerance):
                violations.append("States-TPPI real-component pulse phase must advance 90 degrees per increment")
                break
        receiver0 = rows[0]["receiver"]
        if any(not _close(row["receiver"], receiver0, tolerance) for row in rows):
            violations.append("States-TPPI keeps the receiver phase fixed across acquisitions")

    for increment, members in by_increment.items():
        if "real" in members and "imaginary" in members:
            real_row = members["real"]
            imaginary_row = members["imaginary"]
            if not _close(imaginary_row["phase"] - real_row["phase"], 90.0, tolerance):
                violations.append(f"increment {increment}: imaginary pulse phase must lead real by 90 degrees")
            expected_receiver_shift = 90.0 if mode == "states" else 0.0
            if not _close(imaginary_row["receiver"] - real_row["receiver"], expected_receiver_shift, tolerance):
                violations.append(
                    f"increment {increment}: imaginary receiver shift must be {expected_receiver_shift:g} degrees")

    missing_components = []
    for increment in increments:
        for component in ("real", "imaginary"):
            if component not in by_increment[increment]:
                missing_components.append({"increment": increment, "missing_component": component})
    count = declared_count if declared_count is not None else (max(increments) + 1 if increments else 0)
    missing_increments = [increment for increment in range(count) if increment not in by_increment]
    pair_count = sum(1 for members in by_increment.values()
                     if "real" in members and "imaginary" in members)
    return {
        "mode": mode,
        "declared_increment_count": declared_count,
        "increments": increments,
        "increment_count": count,
        "complete_quadrature": not missing_components and not missing_increments,
        "missing_quadrature_components": missing_components,
        "missing_increments": missing_increments,
        "complete_pairs": pair_count,
        "phase_convention_valid": not violations,
        "phase_violations": violations,
    }


def summarize_quadrature(details):
    """Compact explanation used in the response envelope."""
    if details["mode"] == "none":
        return None
    return {
        "mode": details["mode"],
        "increments": details["increments"],
        "increment_count": details["increment_count"],
        "complete_quadrature": details["complete_quadrature"],
        "complete_pairs": details["complete_pairs"],
        "missing_quadrature_components": details["missing_quadrature_components"],
        "missing_increments": details["missing_increments"],
        "phase_convention_valid": details["phase_convention_valid"],
        "phase_violations": details["phase_violations"],
        "note": "Missing real/imaginary partners are reported, never zero filled",
    }
