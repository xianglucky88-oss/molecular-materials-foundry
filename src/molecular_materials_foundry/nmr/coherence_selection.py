import cmath
import math

def select(channel, phases, receivers, signals):
    coefficients = [cmath.exp(1j*math.radians(receiver-channel*phase))
                    for phase,receiver in zip(phases,receivers)]
    count = len(coefficients)
    result = []
    for column in range(len(signals[0])):
        values = [coefficient*signal[column] for coefficient,signal in zip(coefficients,signals)]
        result.append(complex(math.fsum(z.real for z in values)/count, math.fsum(z.imag for z in values)/count))
    return result
