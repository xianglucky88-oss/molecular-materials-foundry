import math
from ..validation import fields, mapping, sequence, label, integer, number
from ..errors import DomainError

TABLE = {
    "H": ((1.00782503223,0.999885),(2.01410177812,0.000115)),
    "C": ((12.0,0.9893),(13.00335483507,0.0107)),
    "N": ((14.00307400443,0.99636),(15.00010889888,0.00364)),
    "O": ((15.99491461957,0.99757),(16.99913175650,0.00038),(17.99915961286,0.00205)),
    "P": ((30.97376199842,1.0),),
}
ELECTRON_MASS = 0.000548579909065

def prepare(payload):
    fields(payload,["composition"],["populations","charge"])
    composition = mapping(payload["composition"], "composition")
    if not composition or len(composition)>20:
        raise DomainError("composition_size", "Use one to twenty element populations")
    custom = mapping(payload.get("populations",{}), "populations")
    if set(custom)-set(composition):
        raise DomainError("unused_population", "Custom populations must belong to the composition")
    groups = []
    for symbol,count in sorted(composition.items()):
        label(symbol,"element")
        count = integer(count,"atom count",minimum=1,maximum=200)
        source = custom.get(symbol,TABLE.get(symbol))
        if source is None:
            raise DomainError("unknown_element", "No isotope population is available for "+symbol)
        if not isinstance(source,(list,tuple)) or not 1<=len(source)<=8:
            raise DomainError("population_size", "Use one to eight isotopes per population")
        values = []
        for pair in source:
            if not isinstance(pair,(list,tuple)) or len(pair)!=2:
                raise DomainError("invalid_population", "Each isotope is [mass, probability]")
            mass = number(pair[0],"isotope mass",positive=True,maximum=1000)
            probability = number(pair[1],"isotope probability",minimum=0,maximum=1)
            values.append((mass,probability))
        if len({m for m,p in values})!=len(values) or abs(math.fsum(p for m,p in values)-1)>1e-12:
            raise DomainError("invalid_population", "Isotope masses must be distinct and probabilities sum to one")
        groups.append((symbol,count,tuple(values)))
    charge = integer(payload.get("charge",0),"charge",minimum=-20,maximum=20)
    return groups, charge
