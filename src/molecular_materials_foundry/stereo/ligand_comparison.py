from ..validation import fields, sequence, label, integer, number
from ..errors import DomainError

def prepare(payload):
    fields(payload,["atoms","bonds","centers"])
    atoms = {}
    for atom in sequence(payload["atoms"],"atoms",minimum=5,maximum=200):
        fields(atom,["id","atomic_number","position"],["mass_number"])
        name=label(atom["id"],"atom id")
        if name in atoms:
            raise DomainError("duplicate_atom","Atom identifiers must be unique")
        z=integer(atom["atomic_number"],"atomic_number",minimum=1,maximum=118)
        mass=integer(atom.get("mass_number",0),"mass_number",minimum=0,maximum=400)
        pos=tuple(number(x,"coordinate",minimum=-1e6,maximum=1e6) for x in sequence(atom["position"],"position",minimum=3,maximum=3))
        atoms[name]={"z":z,"mass":mass,"position":pos}
    adjacency={name:set() for name in atoms}
    for bond in sequence(payload["bonds"],"bonds",minimum=4,maximum=199):
        pair=sequence(bond,"bond",minimum=2,maximum=2)
        a,b=(label(x,"atom id") for x in pair)
        if a not in atoms or b not in atoms or a==b or b in adjacency[a]:
            raise DomainError("invalid_bond","Bonds must connect distinct known atoms once")
        adjacency[a].add(b);adjacency[b].add(a)
    visited=set()
    def visit(name,parent):
        if name in visited:
            raise DomainError("unsupported_ring","Only acyclic single-bond molecular graphs are supported")
        visited.add(name)
        for neighbor in adjacency[name]-{parent}:
            visit(neighbor,name)
    visit(next(iter(atoms)),None)
    if len(visited)!=len(atoms):
        raise DomainError("disconnected_graph","Use one connected molecular graph")
    centers=sequence(payload["centers"],"centers",minimum=1,maximum=50)
    if any(not isinstance(c,str) or c not in atoms for c in centers) or len(set(centers))!=len(centers):
        raise DomainError("invalid_center","Centers must be distinct known atom identifiers")
    return atoms,adjacency,centers

def priority(atom):
    return atom["z"],atom["mass"]
