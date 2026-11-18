from .acquisition_schedule import prepare
from .coherence_selection import select

def process(payload):
    channel, phases, receivers, signals = prepare(payload)
    result = select(channel,phases,receivers,signals)
    return {"model": "single-axis-uniform-phase-cycle-v1", "channel": channel,
            "alias_modulus": len(phases), "channel_residue": channel%len(phases),
            "receiver_convention": "observed = physical * exp(-i * receiver_phase)",
            "signal": [[value.real,value.imag] for value in result], "acquisition_count": len(phases)}
