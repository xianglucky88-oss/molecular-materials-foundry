from ..validation import fields, sequence, integer, number
from ..errors import DomainError

IDENTITY=((1,0,0),(0,1,0),(0,0,1))

def _multiply(a,b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))

def prepare(payload):
    fields(payload,["operations","observations"],["friedel"])
    if "friedel" in payload and not isinstance(payload["friedel"],bool):
        raise DomainError("invalid_friedel","friedel must be a boolean")
    operations=[]
    for raw in sequence(payload["operations"],"operations",minimum=1,maximum=48):
        rows=sequence(raw,"operation",minimum=3,maximum=3)
        op=tuple(tuple(integer(x,"operation entry",minimum=-2,maximum=2) for x in sequence(row,"row",minimum=3,maximum=3)) for row in rows)
        a,b,c=op
        det=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        if abs(det)!=1:
            raise DomainError("invalid_operation","Operations must be unimodular integer matrices")
        operations.append(op)
    group=set(operations)
    if len(group)!=len(operations) or IDENTITY not in group or any(_multiply(a,b) not in group for a in group for b in group):
        raise DomainError("invalid_group","Operations must be a distinct, complete finite group containing identity")
    observations=[]
    for raw in sequence(payload["observations"],"observations",minimum=1,maximum=10000):
        fields(raw,["hkl","intensity","variance"])
        hkl=tuple(integer(x,"hkl",minimum=-100000,maximum=100000) for x in sequence(raw["hkl"],"hkl",minimum=3,maximum=3))
        if hkl==(0,0,0):
            raise DomainError("direct_beam","The direct beam is not a reflection")
        observations.append((hkl,number(raw["intensity"],"intensity",minimum=-1e100,maximum=1e100),
                             number(raw["variance"],"variance",minimum=1e-100,maximum=1e100)))
    return group,observations,payload.get("friedel",True)

def representative(hkl,operations,friedel):
    orbit={tuple(sum(row[j]*hkl[j] for j in range(3)) for row in op) for op in operations}
    if friedel:
        orbit |= {tuple(-x for x in member) for member in list(orbit)}
    return min(orbit)
