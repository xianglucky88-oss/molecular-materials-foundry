from ..validation import fields, number, integer, sequence
from ..errors import DomainError

def prepare(payload):
    fields(payload, ["acquisitions", "channel"])
    channel = integer(payload["channel"], "channel", minimum=-100, maximum=100)
    acquisitions = sequence(payload["acquisitions"], "acquisitions", minimum=2, maximum=64)
    phases, receivers, signals = [], [], []
    width = None
    for acquisition in acquisitions:
        fields(acquisition, ["phase_degrees", "receiver_degrees", "signal"])
        phases.append(number(acquisition["phase_degrees"], "phase", minimum=-36000, maximum=36000)%360)
        receivers.append(number(acquisition["receiver_degrees"], "receiver", minimum=-36000, maximum=36000)%360)
        samples = sequence(acquisition["signal"], "signal", minimum=1, maximum=10000)
        values = []
        for sample in samples:
            sample = sequence(sample, "complex sample", minimum=2, maximum=2)
            values.append(complex(number(sample[0], "real", minimum=-1e100, maximum=1e100),
                                  number(sample[1], "imaginary", minimum=-1e100, maximum=1e100)))
        if width is not None and width!=len(values):
            raise DomainError("unaligned_acquisitions", "All acquisitions must have equal signal length")
        width = len(values)
        signals.append(values)
    size = len(phases)
    expected = [(phases[0]+360*i/size)%360 for i in range(size)]
    ordered = sorted(phases)
    if any(abs(a-b)>1e-7 for a,b in zip(sorted(expected),ordered)):
        raise DomainError("unsupported_schedule", "Use one complete, uniformly spaced single-axis phase cycle")
    return channel, phases, receivers, signals
