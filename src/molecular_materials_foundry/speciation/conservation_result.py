import math
from .thermo_model import prepare,WATER_KW
from .phase_equilibrium import solve

def process(payload):
    acids,charge=prepare(payload)
    hydrogen,residual=solve(acids,charge)
    species=[]
    for name,total,k in acids:
        ionized=total*k/(k+hydrogen)
        neutral=total*hydrogen/(k+hydrogen)
        species.append({"id":name,"neutral":neutral,"anion":ionized,"total":total,
                        "mass_balance_residual":neutral+ionized-total,
                        "equilibrium_residual":hydrogen*ionized-k*neutral})
    return {"model":"ideal-monoprotic-acids-no-solid-phases-298.15K-v1","hydrogen":hydrogen,
            "hydroxide":WATER_KW/hydrogen,"pH":-math.log10(hydrogen),"strong_ion_charge":charge,
            "charge_balance_residual":residual,"species":species,"status":"converged"}
