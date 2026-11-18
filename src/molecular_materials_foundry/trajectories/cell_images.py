import math
from ..validation import fields, sequence, number, integer, label
from ..errors import DomainError

def prepare(payload):
    fields(payload,["box","atoms","frames"],["max_step","bonds","max_bond_length","lag"])
    box=tuple(number(x,"box length",minimum=1e-9,maximum=1e9) for x in sequence(payload["box"],"box",minimum=3,maximum=3))
    atoms=tuple(label(x,"atom id") for x in sequence(payload["atoms"],"atoms",minimum=1,maximum=1000))
    if len(set(atoms))!=len(atoms):
        raise DomainError("duplicate_atom","Atom identifiers must be distinct")
    frames=[]
    for frame in sequence(payload["frames"],"frames",minimum=1,maximum=1000):
        fields(frame,["positions"],["images"])
        positions=[]
        for row in sequence(frame["positions"],"positions",minimum=len(atoms),maximum=len(atoms)):
            xyz=tuple(number(x,"position",minimum=0) for x in sequence(row,"position",minimum=3,maximum=3))
            if any(value>=length for value,length in zip(xyz,box)):
                raise DomainError("unwrapped_input","Input positions must be wrapped into [0, box length)")
            positions.append(xyz)
        images=None
        if "images" in frame:
            images=[tuple(integer(x,"image",minimum=-10**6,maximum=10**6) for x in sequence(row,"image",minimum=3,maximum=3))
                    for row in sequence(frame["images"],"images",minimum=len(atoms),maximum=len(atoms))]
        frames.append((positions,images))
    if len(frames)*len(atoms)>100000:
        raise DomainError("work_limit","Use at most 100000 frame-atom entries")
    modes={images is not None for positions,images in frames}
    if len(modes)!=1:
        raise DomainError("mixed_image_mode","Every frame must provide images, or every frame must omit them")
    max_step=None
    if False in modes:
        if "max_step" not in payload:
            raise DomainError("ambiguous_images","Inferred images require an explicit maximum displacement assumption")
        max_step=number(payload["max_step"],"max_step",minimum=0)
        if max_step>=min(box)/2:
            raise DomainError("ambiguous_images","Maximum displacement must be strictly less than half the shortest box side")
    elif "max_step" in payload:
        raise DomainError("unused_assumption","max_step applies only to trajectories without explicit images")
    lag=integer(payload.get("lag",1),"lag",minimum=1,maximum=1000)
    return box,atoms,frames,max_step,lag

def reconstruct(box,frames,max_step):
    result=[]
    for frame_index,(positions,images) in enumerate(frames):
        if images is not None:
            current=[tuple(value+image*length for value,image,length in zip(row,flags,box)) for row,flags in zip(positions,images)]
        elif not result:
            current=list(positions)
        else:
            previous_wrapped=frames[frame_index-1][0]
            current=[]
            for atom,(position,previous) in enumerate(zip(positions,previous_wrapped)):
                delta=tuple((value-old)-length*math.floor((value-old)/length+0.5) for value,old,length in zip(position,previous,box))
                if math.sqrt(math.fsum(x*x for x in delta))>max_step+1e-12*max_step+32*math.ulp(min(box)):
                    raise DomainError("step_bound_violation","No periodic image satisfies the declared displacement bound")
                current.append(tuple(old+change for old,change in zip(result[-1][atom],delta)))
        result.append(current)
    return result
