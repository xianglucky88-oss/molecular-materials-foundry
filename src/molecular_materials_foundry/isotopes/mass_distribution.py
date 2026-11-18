import math
from ..errors import DomainError

def compositions(total,size):
    if size==1:
        yield (total,)
    else:
        for value in range(total+1):
            for rest in compositions(total-value,size-1):
                yield (value,)+rest

def enumerate_distribution(groups):
    states = [(0.0,1.0)]
    for symbol,count,isotopes in groups:
        size = math.comb(count+len(isotopes)-1,len(isotopes)-1)
        if size*len(states)>100000:
            raise DomainError("state_limit", "Complete isotopic composition enumeration exceeds 100000 states")
        elemental = []
        for allocation in compositions(count,len(isotopes)):
            if any(n and p==0 for n,(mass,p) in zip(allocation,isotopes)):
                continue
            mass = math.fsum(n*m for n,(m,p) in zip(allocation,isotopes))
            log_p = math.lgamma(count+1)-math.fsum(math.lgamma(n+1) for n in allocation)
            log_p += math.fsum(n*math.log(p) for n,(mass,p) in zip(allocation,isotopes) if n)
            probability = math.exp(log_p)
            if probability == 0:
                raise DomainError("probability_range", "An isotopic state's probability is not representable")
            elemental.append((mass,probability))
        states = [(left+right,p*q) for left,p in states for right,q in elemental]
        if any(p==0 for mass,p in states):
            raise DomainError("probability_range", "A combined isotopic state's probability is not representable")
    return states
