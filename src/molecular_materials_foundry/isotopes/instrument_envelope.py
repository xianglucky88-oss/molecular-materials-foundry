import math
from .populations import prepare, ELECTRON_MASS
from .mass_distribution import enumerate_distribution
from ..errors import DomainError

def process(payload):
    groups, charge = prepare(payload)
    states = enumerate_distribution(groups)
    merged = {}
    for mass,probability in states:
        ion_mass = mass-charge*ELECTRON_MASS
        if ion_mass <= 0:
            raise DomainError("invalid_ion_mass", "The declared charge produces a nonpositive ion mass")
        coordinate = ion_mass/abs(charge) if charge else ion_mass
        merged.setdefault(coordinate,[]).append(probability)
    peaks = [{"coordinate": mass,"probability": math.fsum(probabilities)} for mass,probabilities in sorted(merged.items())]
    return {"model": "complete-composition-enumeration-v1", "coordinate": "mass_to_charge" if charge else "neutral_mass",
            "charge": charge, "state_count": len(states), "peaks": peaks,
            "probability_sum": math.fsum(p["probability"] for p in peaks), "truncated_probability": 0.0,
            "electron_mass_u": ELECTRON_MASS}
