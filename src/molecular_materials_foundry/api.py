from .errors import DomainError
from .validation import mapping
from .nmr.hypercomplex_dataset import process as nmr
from .isotopes.instrument_envelope import process as isotopes
from .stereo.descriptors import process as stereo
from .reflections.merged_intensity import process as reflections
from .speciation.conservation_result import process as speciation
from .trajectories.displacement_evidence import process as trajectories

ROUTES={"/nmr":nmr,"/isotopes":isotopes,"/stereo":stereo,"/reflections":reflections,
        "/speciation":speciation,"/trajectories":trajectories}

def dispatch(path,payload):
    if path not in ROUTES:
        raise DomainError("unknown_endpoint","Unknown endpoint")
    return ROUTES[path](mapping(payload))
