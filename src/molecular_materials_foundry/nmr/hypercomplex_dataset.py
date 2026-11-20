"""Domain entry point for coherence-channel selection and quantification."""
from .acquisition_schedule import prepare
from .coherence_selection import select
from .estimation import estimate
from .assembly import assemble


def process(payload):
    prepared = prepare(payload)
    if prepared[0] == "legacy":
        _, channel, phases, receivers, signals = prepared
        result = select(channel, phases, receivers, signals)
        return {"model": "single-axis-uniform-phase-cycle-v1", "channel": channel,
                "alias_modulus": len(phases), "channel_residue": channel % len(phases),
                "receiver_convention": "observed = physical * exp(-i * receiver_phase)",
                "signal": [[value.real, value.imag] for value in result],
                "acquisition_count": len(phases)}
    _, model = prepared
    estimation = estimate(model["basis"], model["rows"], model["candidates"])
    return assemble(model, estimation)
