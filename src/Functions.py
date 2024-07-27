import math

def binarySigmoid(x: float):
    return 1 / (1 + pow(math.e, -x))

def dBinarySigmoid(x: float):
    return binarySigmoid(x) * (1 - binarySigmoid(x))

def bipolarSigmoid(x: float):
    r = 0
    
    try:
        r = -1 + 2 / (1 + pow(math.e, -x))
    except:
        None
    
    return r

def dBipolarSigmoid(x: float):
    return 0.5 * (1 - pow(bipolarSigmoid(x), 2))

def average(a: list[float]):
    m = 0
    
    for x in a:
        m += x
    
    return m / len(a)
