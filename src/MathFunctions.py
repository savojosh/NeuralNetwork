import math

def binaryStep(x: float, derivative: bool=False) -> float:
    """
    **Parameters**
    - *x: float*
    > Input
    - *derivative: bool=False*
    > Calculate derivative. 
    """

    if(derivative):
        return 0
    
    return (0 if x < 0 else 1)

def sigmoid(x: float, derivative: bool=False) -> float:
    """
    **Parameters**
    - *x: float*
    > Input
    - *derivative: bool=False*
    > Calculate derivative. 
    """
    
    if(derivative):
        return sigmoid(x) * (1 - sigmoid(x))
    
    return 1 / (1 + pow(math.e, -x))

def tanh(x: float, derivative: bool=False) -> float:
    """
    **Parameters**
    - *x: float*
    > Input
    - *derivative: bool=False*
    > Calculate derivative.
    """

    if(derivative):
        return 1 - pow(tanh(x, derivative=False), 2)
            # TODO: Derivative for the smht is wrong here.
    
    return (pow(math.e, x) - pow(math.e, -x)) / (pow(math.e, x) + pow(math.e, -x))

def mse(x: float, y: float, derivative=False):
    """
    **Parameters**
    - *x: float*
    > Real output
    - *y: float*
    > Expected output
    - *derivative: bool=False*
    > Calculate derivative.
    """

    if(derivative):
        return 2 * (x - y)
    
    return pow(x - y, 2)

def average(a: list[float]) -> float:
    """
    **Parameters**
    - *a: list[float]*
    > Input array
    """
    
    m = 0
    
    for x in a:
        m += x
    
    return m / len(a)

def floorToDecimal(x: float, place: int=10) -> float:

    return math.floor(x * place) / place

def vectorAddition(a: list[int]|list[float], b: list[int]|list[float]) -> list:

    c = []
    for x, y in zip(a, b):
        c.append(x + y)

    return c

def vectorSubtraction(a: list[int]|list[float], b: list[int]|list[float]) -> list:

    c = []
    for x, y in zip(a, b):
        c.append(x - y)

    return c
