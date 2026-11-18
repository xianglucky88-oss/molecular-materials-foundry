import math
from ..validation import number, sequence, label
from ..errors import DomainError

def validate_bonds(payload,atoms,continuous):
    bonds=sequence(payload.get("bonds",[]),"bonds",maximum=10000)
    if bool(bonds)!=("max_bond_length" in payload):
        raise DomainError("bond_contract","Nonempty bonds and max_bond_length must be supplied together")
    if not bonds:
        return
    limit=number(payload["max_bond_length"],"max_bond_length",positive=True,maximum=1e12)
    indices={name:i for i,name in enumerate(atoms)}
    seen=set()
    for raw in bonds:
        raw=sequence(raw,"bond",minimum=2,maximum=2)
        names=[label(x,"bond atom") for x in raw]
        if any(name not in indices for name in names) or names[0]==names[1]:
            raise DomainError("invalid_bond","Bond references must name distinct known atoms")
        key=tuple(sorted(names))
        if key in seen:
            raise DomainError("duplicate_bond","Each bond must be specified once")
        seen.add(key)
        a,b=(indices[name] for name in names)
        for frame in continuous:
            distance=math.sqrt(math.fsum((x-y)**2 for x,y in zip(frame[a],frame[b])))
            if distance>limit:
                raise DomainError("bond_image_conflict","Unwrapped molecular coordinates violate the declared bond bound")
