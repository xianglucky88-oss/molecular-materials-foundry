import math
from .thermo_model import WATER_KW
from ..errors import DomainError

def solve(acids,strong_charge):
    def charge_balance(hydrogen):
        return hydrogen+strong_charge-WATER_KW/hydrogen-math.fsum(total*k/(k+hydrogen) for name,total,k in acids)
    lower,upper=-18.0,2.0
    if charge_balance(10**lower)>0 or charge_balance(10**upper)<0:
        raise DomainError("unsupported_range","Charge balance lies outside the supported hydrogen range")
    for iteration in range(160):
        middle=(lower+upper)/2
        if charge_balance(10**middle)>0:
            upper=middle
        else:
            lower=middle
    hydrogen=10**((lower+upper)/2)
    return hydrogen,charge_balance(hydrogen)
