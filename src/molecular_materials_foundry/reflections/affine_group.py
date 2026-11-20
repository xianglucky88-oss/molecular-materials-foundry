"""Affine space-group operations in reciprocal space and basis changes.

Conventions
-----------
Reflection indices are row vectors ``h = (h1, h2, h3)``.  An affine operation
is ``(R, t)`` with an integer unimodular 3x3 matrix ``R`` and a translation
``t`` given in twelfths of a lattice vector.  A reflection is mapped to
``h R`` and acquires the phase factor ``exp(2*pi*i * h . t)``, so the
reflection is a systematic absence whenever the operation maps it onto
itself (or onto its Friedel opposite) with a non-integer translation dot
product.

Twelfths exactly represent every translation component used by the
crystallographic space groups (denominators 1, 2, 3, 4, 6, 12).
"""
from fractions import Fraction
from ..validation import fields, sequence, integer, number
from ..errors import DomainError

IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_TWELFTH_LIMIT = 1e-9


def multiply(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def determinant(matrix):
    a, b, c = matrix
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def integer_unimodular_inverse(matrix):
    """Inverse of an integer unimodular 3x3 matrix, as integer tuples."""
    a, b, c = matrix
    cofactors = (
        (b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]),
        (a[2] * c[1] - a[1] * c[2], a[0] * c[2] - a[2] * c[0], a[1] * c[0] - a[0] * c[1]),
        (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]),
    )
    det = determinant(matrix)
    if det not in (1, -1):
        raise DomainError("invalid_operation", "Operations must be unimodular integer matrices")
    return tuple(tuple(cofactors[j][i] // det for j in range(3)) for i in range(3))


def _parse_matrix(raw, name, bound):
    rows = sequence(raw, name, minimum=3, maximum=3)
    matrix = tuple(
        tuple(integer(x, name + " entry", minimum=-bound, maximum=bound)
              for x in sequence(row, "row", minimum=3, maximum=3))
        for row in rows
    )
    if abs(determinant(matrix)) != 1:
        raise DomainError("invalid_operation", "Operations must be unimodular integer matrices")
    return matrix


def _parse_translation(raw, name):
    values = sequence(raw, name, minimum=3, maximum=3)
    result = []
    for value in values:
        probe = number(value, name, minimum=-1.0, maximum=1.0)
        twelfths = round(probe * 12.0)
        if abs(probe - twelfths / 12.0) > _TWELFTH_LIMIT:
            raise DomainError(
                "invalid_translation",
                name + " components must be multiples of 1/12 of a lattice vector",
            )
        result.append(Fraction(twelfths, 12))
    return tuple(result)


def parse_operation(raw):
    fields(raw, ["rotation", "translation"], [])
    rotation = _parse_matrix(raw["rotation"], "rotation", 2)
    translation = _parse_translation(raw["translation"], "translation")
    return rotation, translation


def canonical(operation):
    rotation, translation = operation
    return rotation, tuple(component % 1 for component in translation)


def compose(first, second):
    """Affine composition: apply ``first`` then ``second`` (row-vector order)."""
    r1, t1 = first
    r2, t2 = second
    rotation = multiply(r1, r2)
    translation = tuple(t2[j] + sum(t1[k] * r2[k][j] for k in range(3)) for j in range(3))
    return canonical((rotation, translation))


def _rotation_group(operations):
    return {operation[0] for operation in operations}


def validate_affine_group(operations):
    """Check closure (including inverses) and the identity operation."""
    canon = [canonical(operation) for operation in operations]
    keys = set(canon)
    if len(keys) != len(canon):
        raise DomainError("invalid_group", "Operations must be distinct")
    identity = (IDENTITY, (Fraction(0),) * 3)
    if identity not in keys:
        raise DomainError("invalid_group", "The identity operation (zero translation) is required")
    for first in keys:
        for second in keys:
            if compose(first, second) not in keys:
                raise DomainError("invalid_group", "Operations must form a complete finite affine group")
    # Closure plus the identity already guarantees a finite inverse, but verify it explicitly
    # so a malformed request cannot rely on the subgroup argument implicitly.
    for rotation, translation in keys:
        inverse_rotation = integer_unimodular_inverse(rotation)
        inverse_translation = tuple(-sum(translation[k] * inverse_rotation[j][k] for k in range(3))
                                    for j in range(3))
        if canonical((inverse_rotation, inverse_translation)) not in keys:
            raise DomainError("invalid_group", "Every operation must have its inverse in the group")
    return keys


def parse_basis_change(raw):
    matrix = _parse_matrix(raw, "basis_change", 8)
    if determinant(matrix) == -1:
        # A handedness-reversing index map cannot describe the same lattice.
        raise DomainError("invalid_basis_change", "basis_change must preserve handedness (determinant +1)")
    return matrix


def conjugate_operation(operation, basis_change):
    """Move an operation from a batch frame into the common reference frame.

    Batch indices satisfy ``h_ref = h_batch P``.  An operation ``(R, t)`` in
    the batch frame becomes ``(P^-1 R P, P^-1 t)`` in the reference frame.
    """
    rotation, translation = operation
    p_inverse = integer_unimodular_inverse(basis_change)
    new_rotation = multiply(multiply(p_inverse, rotation), basis_change)
    new_translation = tuple(sum(translation[k] * p_inverse[j][k] for k in range(3)) for j in range(3))
    return canonical((new_rotation, new_translation))


def transform_index(hkl, basis_change):
    return tuple(sum(hkl[j] * basis_change[j][k] for j in range(3)) for k in range(3))


def common_reference_group(batch_groups, basis_changes):
    """Conjugate every batch group and require one common affine group."""
    conjugated = []
    for group, basis_change in zip(batch_groups, basis_changes):
        moved = frozenset(conjugate_operation(operation, basis_change) for operation in group)
        conjugated.append(moved)
    reference = conjugated[0]
    for moved in conjugated[1:]:
        if moved != reference:
            raise DomainError(
                "incompatible_space_groups",
                "Batches do not describe the same space group after their basis changes",
            )
    return sorted(reference, key=lambda item: (item[0], tuple(round(float(x), 6) for x in item[1])))


def extinction_reason(hkl, operations, preserve_friedel):
    """Return the translation responsible for extinction, or None.

    A reflection is systematically absent when an operation maps it onto
    itself (or, for an inversion-free crystal where Friedel opposites are
    kept, onto its Friedel opposite) and the corresponding phase is not
    unity.
    """
    for rotation, translation in operations:
        mapped = tuple(sum(row[j] * hkl[j] for j in range(3)) for row in rotation)
        if mapped == hkl or (preserve_friedel and mapped == tuple(-x for x in hkl)):
            phase = sum(hkl[j] * translation[j] for j in range(3))
            if phase.denominator != 1:
                return rotation, translation
    return None


def translation_as_text(translation):
    return [f"{component.numerator}/{component.denominator}" if component.denominator != 1
            else str(component.numerator) for component in translation]
