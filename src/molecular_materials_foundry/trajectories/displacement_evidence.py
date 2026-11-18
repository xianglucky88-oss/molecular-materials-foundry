import math
from .cell_images import prepare,reconstruct
from .molecular_continuity import validate_bonds

def process(payload):
    box,atoms,frames,max_step,lag=prepare(payload)
    continuous=reconstruct(box,frames,max_step)
    validate_bonds(payload,atoms,continuous)
    windows=[]
    for start in range(max(0,len(continuous)-lag)):
        finish=start+lag
        squared=[math.fsum((b-a)**2 for a,b in zip(left,right))
                 for left,right in zip(continuous[start],continuous[finish])]
        windows.append({"start_frame":start,"end_frame":finish,"squared_displacements":squared,
                        "mean_squared_displacement":math.fsum(squared)/len(atoms)})
    return {"model":"fixed-orthorhombic-cell-v1","atoms":list(atoms),"continuous_positions":continuous,
            "image_mode":"explicit" if max_step is None else "bounded_step_assumption",
            "maximum_step_assumption":max_step,"lag_frames":lag,"windows":windows}
