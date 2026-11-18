from .ligand_comparison import prepare,priority
from .dependent_units import orientation, contains_center

def process(payload):
    atoms,adjacency,centers=prepare(payload)
    results=[]
    for center in centers:
        neighbors=adjacency[center]
        if len(neighbors)!=4:
            results.append({"center":center,"descriptor":None,"status":"unsupported_coordination"})
            continue
        keys=[priority(atoms[name]) for name in neighbors]
        if len(set(keys))!=4:
            results.append({"center":center,"descriptor":None,"status":"requires_deeper_ligand_comparison"})
            continue
        by_element={}
        mixed=False
        for name in neighbors:
            atom=atoms[name];by_element.setdefault(atom["z"],[]).append(atom["mass"])
        for masses in by_element.values():
            if len(masses)>1 and 0 in masses:
                mixed=True
        if mixed:
            results.append({"center":center,"descriptor":None,"status":"isotope_unspecified"})
            continue
        ranked=sorted(neighbors,key=lambda name:priority(atoms[name]),reverse=True)
        positions=[atoms[name]["position"] for name in ranked]
        value=orientation(positions)
        if value and not contains_center(atoms[center]["position"], positions):
            results.append({"center":center,"descriptor":None,"status":"inconsistent_tetrahedral_center"})
            continue
        results.append({"center":center,"descriptor":value,"status":"ok" if value else "degenerate_geometry",
                        "priority_order":ranked})
    return {"model":"acyclic-single-bond-first-sphere-CIP-subset-v1","implicit_hydrogens":False,"centers":results}
