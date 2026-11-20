"""Small statistical helpers shared by the multi-batch reflection pipeline.

Only the Python standard library is used.
"""
import math


def chi_survival(value, degrees_of_freedom):
    """Upper-tail probability P(X >= value) for a chi-squared variable.

    Returns None when no test is defined (zero degrees of freedom).
    """
    if degrees_of_freedom is None or degrees_of_freedom <= 0:
        return None
    if value <= 0.0:
        return 1.0
    half = value / 2.0
    if degrees_of_freedom % 2 == 0:
        # Q(value, 2) = exp(-value/2)
        q = math.exp(-half)
        nu = 2
    else:
        # Q(value, 1) = erfc(sqrt(value/2))
        q = math.erfc(math.sqrt(half))
        nu = 1
    # Recurrence Q(value, nu+2) = Q(value, nu) + (value/2)**(nu/2) e**(-value/2) / Gamma(nu/2+1)
    while nu < degrees_of_freedom:
        log_term = -half + (nu / 2.0) * math.log(half) - math.lgamma(nu / 2.0 + 1.0)
        q += math.exp(log_term)
        nu += 2
    return min(1.0, max(0.0, q))


def tension_status(p_value):
    if p_value is None:
        return "not_estimable"
    return "tension" if p_value < 0.05 else "ok"
