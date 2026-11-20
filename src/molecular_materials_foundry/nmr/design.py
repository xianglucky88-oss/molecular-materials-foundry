"""Structural analysis of the complex channel design matrix.

The acquisition model is

    z_j = sum_q x_jq * a_q + noise_j,  x_jq = exp(i * q * phase_j)

after receiver-phase correction.  All routines use only the standard
library so the service keeps its zero-dependency runtime.
"""
import cmath
import math

EPS = 2.220446049250313e-16
ALIAS_TOLERANCE = 1e-9
RANK_FLOOR = 1e-12


def identity(size):
    return [[1.0 + 0.0j if row == col else 0.0j for col in range(size)] for row in range(size)]


def hermitian_eigen(matrix, max_sweeps=100):
    """Cyclic Jacobi eigendecomposition of a Hermitian matrix.

    Uses the standard two-angle Hermitian rotation.  Returns
    ``(eigenvalues, eigenvectors)`` with real eigenvalues sorted ascending and
    ``eigenvectors[row][k]`` the k-th unit eigenvector component.
    """
    size = len(matrix)
    work = [row[:] for row in matrix]
    vectors = identity(size)
    for _ in range(max_sweeps):
        off_diagonal = math.fsum(abs(work[p][q]) ** 2
                                 for p in range(size) for q in range(p + 1, size))
        if off_diagonal <= (EPS ** 2) * size * size:
            break
        for p in range(size):
            for q in range(p + 1, size):
                apq = work[p][q]
                if abs(apq) <= EPS:
                    continue
                app = work[p][p].real
                aqq = work[q][q].real
                if abs(aqq - app) <= EPS:
                    theta = math.pi / 4.0
                else:
                    theta = 0.5 * math.atan2(2.0 * abs(apq), aqq - app)
                c = math.cos(theta)
                s = math.sin(theta)
                phase = cmath.exp(-1j * cmath.phase(apq))
                # Off-pivot rows and columns; pivots are set explicitly.
                app_new = c * c * app + s * s * aqq - 2.0 * c * s * abs(apq)
                aqq_new = s * s * app + c * c * aqq + 2.0 * c * s * abs(apq)
                for k in range(size):
                    if k == p or k == q:
                        continue
                    akp = work[k][p]
                    akq = work[k][q]
                    work[k][p] = c * akp - s * phase * akq
                    work[k][q] = s * phase.conjugate() * akp + c * akq
                    work[p][k] = work[k][p].conjugate()
                    work[q][k] = work[k][q].conjugate()
                work[p][p] = complex(app_new)
                work[q][q] = complex(aqq_new)
                work[p][q] = 0.0j
                work[q][p] = 0.0j
                for k in range(size):
                    vkp = vectors[k][p]
                    vkq = vectors[k][q]
                    vectors[k][p] = c * vkp - s * phase * vkq
                    vectors[k][q] = s * phase.conjugate() * vkp + c * vkq
    eigenvalues = [work[k][k].real for k in range(size)]
    order = sorted(range(size), key=lambda k: eigenvalues[k])
    ordered = [eigenvalues[k] for k in order]
    eigenvectors = [[vectors[row][k] for k in order] for row in range(size)]
    return ordered, eigenvectors


def invert_hermitian(matrix):
    """Gauss-Jordan inverse with partial pivoting for complex Hermitian input."""
    size = len(matrix)
    unit = identity(size)
    work = [matrix[row][:] + unit[row] for row in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(work[row][col]))
        if abs(work[pivot][col]) <= RANK_FLOOR:
            raise ValueError("singular_matrix")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
        pivot_value = work[col][col]
        work[col] = [entry / pivot_value for entry in work[col]]
        for row in range(size):
            if row == col:
                continue
            factor = work[row][col]
            if factor == 0.0j:
                continue
            work[row] = [value - factor * pivot_entry for value, pivot_entry in zip(work[row], work[col])]
    return [row[size:] for row in work]


def gram(basis, weights=None):
    """Weighted Gram matrix ``basis^H diag(weights) basis``."""
    rows = len(basis)
    size = len(basis[0])
    if weights is None:
        weights = [1.0] * rows
    result = [[0.0j for _ in range(size)] for _ in range(size)]
    for p in range(size):
        for q in range(p, size):
            total = 0.0j
            for j in range(rows):
                total += weights[j] * basis[j][p].conjugate() * basis[j][q]
            result[p][q] = total
            result[q][p] = total.conjugate()
    return result


def structure(basis):
    """Rank, conditioning and exact alias classes of the candidate design.

    Aliasing is structural: two candidate columns are aliased when their
    normalized inner product has unit modulus, i.e. every observed
    acquisition multiplies them by the same phase factor.
    """
    size = len(basis[0])
    matrix = gram(basis)
    eigenvalues, _ = hermitian_eigen(matrix)
    largest = max(eigenvalues)
    threshold = max(RANK_FLOOR, size * EPS) * largest
    rank = sum(value > threshold for value in eigenvalues)
    positive = [value for value in eigenvalues if value > threshold]
    condition = positive[0] and positive[-1] / positive[0] if positive else None
    aliases = [set() for _ in range(size)]
    for p in range(size):
        diagonal_p = matrix[p][p].real
        for q in range(p + 1, size):
            diagonal_q = matrix[q][q].real
            if diagonal_p <= 0.0 or diagonal_q <= 0.0:
                continue
            overlap = abs(matrix[p][q]) / math.sqrt(diagonal_p * diagonal_q)
            if overlap >= 1.0 - ALIAS_TOLERANCE:
                aliases[p].add(q)
                aliases[q].add(p)
    groups = _alias_groups(aliases)
    accounted_deficiency = sum(len(group) - 1 for group in groups)
    generic_deficiency = (size - rank) > accounted_deficiency
    return {"rank": rank, "condition_number": condition, "alias_groups": groups,
            "generic_rank_deficiency": generic_deficiency, "eigenvalues": eigenvalues}


def _alias_groups(aliases):
    seen = set()
    groups = []
    for start in range(len(aliases)):
        if start in seen or not aliases[start]:
            continue
        stack = [start]
        group = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            group.append(node)
            stack.extend(aliases[node] - seen)
        groups.append(sorted(group))
    return groups


def alias_factor(basis, target, other):
    """Phase factor relating two aliased columns: x_target = factor * x_other."""
    for row in basis:
        if abs(row[other]) > EPS:
            return row[target] / row[other]
    return None
