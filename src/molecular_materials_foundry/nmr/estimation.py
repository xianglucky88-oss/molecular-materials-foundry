"""Weighted complex least-squares estimation of coherence channels.

Receiver-corrected measurements follow

    y_j = exp(i * receiver_j) * z_j = sum_q exp(i * q * phase_j) * a_q + e_j

with independent, circular-complex noise.  Missing acquisitions are simply
rows absent from the design; zero filling never happens here.  Long time
domains are processed column by column in bounded blocks, and Gram-matrix
factorizations are cached whenever columns share a variance profile.
"""
import math

from . import design as design_module

DEFAULT_BLOCK_SIZE = 128
_GRAM_FLOOR = 1e-14


def weighted_gram(basis, weights):
    """Hermitian ``basis^H diag(weights) basis``."""
    rows = len(basis)
    size = len(basis[0])
    result = [[0.0j for _ in range(size)] for _ in range(size)]
    for p in range(size):
        for q in range(p, size):
            total = 0.0j
            for j in range(rows):
                total += weights[j] * basis[j][p].conjugate() * basis[j][q]
            result[p][q] = total
            result[q][p] = total.conjugate()
    return result


def _row_variance(variance, column):
    if variance is None:
        return None
    return variance[min(column, len(variance) - 1)]


def _profile_key(rows, column, all_present):
    if not all_present:
        return ()
    return tuple(_row_variance(row["variance"], column) for row in rows)


def _independent_indices(basis):
    """Pivoted, numerically deflated selection of independent columns."""
    size = len(basis[0])
    gram = weighted_gram(basis, [1.0] * len(basis))
    selected = []
    for _ in range(size):
        candidates = [index for index in range(size) if index not in selected]
        pivot = max(candidates, key=lambda index: gram[index][index].real)
        pivot_norm = gram[pivot][pivot].real
        if pivot_norm <= _GRAM_FLOOR:
            break
        if not selected:
            selected.append(pivot)
            continue
        sub = [[gram[p][q] for q in selected] for p in selected]
        inverse = design_module.invert_hermitian(sub)
        cross = [gram[pivot][q] for q in selected]
        projected = sum(cross[k].conjugate() * inverse[k][l] * cross[l]
                        for k in range(len(selected)) for l in range(len(selected)))
        if pivot_norm - projected.real <= 1e-12 * max(pivot_norm, 1.0):
            break
        selected.append(pivot)
    selected = sorted(selected)
    return selected, [index for index in range(size) if index not in selected]


def _condition(gram):
    if not gram:
        return None
    eigenvalues, _ = design_module.hermitian_eigen(gram)
    largest = max(abs(value) for value in eigenvalues)
    positive = [value for value in eigenvalues if value > _GRAM_FLOOR * largest]
    if not positive:
        return None
    return positive[-1] / positive[0]


def estimate(basis, rows, candidates, block_size=DEFAULT_BLOCK_SIZE):
    """Estimate every identifiable candidate channel.

    Returns structural diagnostics plus, per identifiable candidate index,
    the complex estimate and the unscaled Gram diagonal used to build
    confidence information during assembly.
    """
    width = len(rows[0]["mean"])
    structure = design_module.structure(basis)
    pivoted, pivot_dependent = _independent_indices(basis)
    aliased = {member for group in structure["alias_groups"] for member in group}
    # Every member of a structural alias class is unidentifiable, including the
    # pivot that numerical column selection happened to retain.
    independent = [index for index in pivoted if index not in aliased]
    dependent = sorted(set(pivot_dependent) | (set(pivoted) - set(independent)))
    sub_basis = [[row[index] for index in independent] for row in basis]
    all_present = all(row["variance"] is not None for row in rows)
    any_present = any(row["variance"] is not None for row in rows)

    records = {index: {"estimate": [None] * width, "gram_diagonal": [None] * width,
                       "scale": [None] * width, "degrees_of_freedom": [None] * width}
               for index in independent}
    uniform = [1.0] * len(rows)

    if independent:
        for block_start in range(0, width, block_size):
            factorizations = {}
            for column in range(block_start, min(width, block_start + block_size)):
                key = _profile_key(rows, column, all_present)
                if all_present:
                    weights = [1.0 / value for value in key]
                else:
                    weights = uniform
                if key not in factorizations:
                    gram = weighted_gram(sub_basis, weights)
                    factorizations[key] = (design_module.invert_hermitian(gram), _condition(gram))
                inverse, _ = factorizations[key]

                corrected = []
                for row in rows:
                    observed = row["mean"][column]
                    corrected.append(observed * complex(math.cos(math.radians(row["receiver"])),
                                                        math.sin(math.radians(row["receiver"]))))
                rhs = []
                for k in range(len(independent)):
                    total = 0.0j
                    for j, row in enumerate(sub_basis):
                        total += weights[j] * row[k].conjugate() * corrected[j]
                    rhs.append(total)
                estimates = [sum(inverse[k][l] * rhs[l] for l in range(len(independent)))
                             for k in range(len(independent))]

                scale, degrees_of_freedom = 1.0, None
                if not all_present:
                    residual_square = 0.0
                    for j, row in enumerate(sub_basis):
                        fitted = sum(row[k] * estimates[k] for k in range(len(independent)))
                        residual = corrected[j] - fitted
                        residual_square += weights[j] * (residual.real ** 2 + residual.imag ** 2)
                    degrees_of_freedom = len(rows) - len(independent)
                    scale = residual_square / degrees_of_freedom if degrees_of_freedom > 0 else None
                for position, index in enumerate(independent):
                    records[index]["estimate"][column] = estimates[position]
                    records[index]["gram_diagonal"][column] = inverse[position][position].real
                    records[index]["scale"][column] = scale
                    records[index]["degrees_of_freedom"][column] = degrees_of_freedom

    full_gram = weighted_gram(basis, uniform)
    return {"structure": structure, "independent": independent, "dependent": dependent,
            "records": records, "design_condition_number": _condition(full_gram),
            "weighted": all_present, "partial_variance": any_present and not all_present}
