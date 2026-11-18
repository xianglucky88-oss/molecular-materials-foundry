from ..validation import fields, sequence, number, label
from ..errors import DomainError

WATER_KW=1e-14

def prepare(payload):
    fields(payload,["acids"],["strong_ion_charge","temperature_k"])
    temperature=number(payload.get("temperature_k",298.15),"temperature_k")
    if temperature!=298.15:
        raise DomainError("unsupported_temperature","The ideal aqueous model is fixed at 298.15 K")
    acids=[];seen=set()
    for raw in sequence(payload["acids"],"acids",maximum=32):
        fields(raw,["id","total","pka"])
        name=label(raw["id"],"acid id")
        if name in seen:
            raise DomainError("duplicate_acid","Acid identifiers must be unique")
        seen.add(name)
        total=number(raw["total"],"total",minimum=0,maximum=1)
        pka=number(raw["pka"],"pka",minimum=-2,maximum=16)
        acids.append((name,total,10**(-pka)))
    charge=number(payload.get("strong_ion_charge",0),"strong_ion_charge",minimum=-1,maximum=1)
    return acids,charge
