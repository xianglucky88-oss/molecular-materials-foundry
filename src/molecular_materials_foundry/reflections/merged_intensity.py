from .symmetry_orbits import prepare,representative
from .measurement_weights import merge

def process(payload):
    operations,observations,friedel=prepare(payload)
    groups={}
    for index,(hkl,intensity,variance) in enumerate(observations):
        key=representative(hkl,operations,friedel)
        groups.setdefault(key,[]).append((intensity,variance,index))
    results=[]
    for hkl,values in sorted(groups.items()):
        intensity,variance,chi_squared=merge(values)
        results.append({"hkl":list(hkl),"intensity":intensity,"variance":variance,
                        "chi_squared":chi_squared,"degrees_of_freedom":len(values)-1,
                        "observation_indices":[v[2] for v in values]})
    return {"model":"explicit-point-group-independent-errors-v1","friedel_merged":friedel,
            "reflections":results,"systematic_absences_evaluated":False}
