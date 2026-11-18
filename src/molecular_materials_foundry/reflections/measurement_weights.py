import math

def merge(values):
    scale=min(variance for _,variance,_ in values)
    weights=[scale/variance for _,variance,_ in values]
    total=math.fsum(weights)
    intensity=math.fsum(value*(weight/total) for (value,variance,index),weight in zip(values,weights))
    variance=scale/total
    chi_squared=math.fsum((value-intensity)**2/uncertainty for value,uncertainty,index in values)
    return intensity,variance,chi_squared
