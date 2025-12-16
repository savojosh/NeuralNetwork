import math
import random
import threading
from pandas import DataFrame
from enum import Enum
from math import isinf, isnan
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

class _Prefix(Enum):
    NODE = "n"
    BIAS = "b"
    WEIGHT = "w"

class Layer:

    #-------------[ CONSTRUCTORS ]-------------#
    
    def __init__(
        self, size: int, inputSize: int, 
        biases: list[float]=None, weights: list[list[float]]=None, 
        lower: float=0, upper: float=1
    ) -> object:
        """
        **Parameters**
        - *size: int*
        > Size of the layer. 
        > Determines the number of nodes.
        - *inputSize: int*
        > Number of inputs feeding into this layer. 
        > Determines the number of weights per node.
        - *biases: list[float]=None*
        > Optional.
        > Biases to build this layer from.
        > Length should match the size parameter.
        - *weights: list[float]=None*
        > Optional.
        > Weights to build this layer from.
        > Length should match the size parameter, and the width should match the inputSize parameter.
        """
                
        if(biases is None): self.biases = [] 
        else: self.biases = biases
        
        if(weights is None): self.weights = []
        else: self.weights = weights

        self.biasesGradient = []
        self.weightsGradient = []
    
        for n in range(size):
            
            if(biases is None): self.biases.append(0.0)
            if(weights is None): self.weights.append([])

            self.biasesGradient.append(0.0)
            self.weightsGradient.append([])
            
            for w in range(inputSize):
                
                if(weights is None): self.weights[n].append(random.uniform(lower, upper))

                self.weightsGradient[n].append(0.0)

    @classmethod
    def fromDataFrame(cls, df: DataFrame) -> object:
        """
        Nodes should be expressed as columns in the DataFrame.
        """
        
        biases = []
        weights = []
        for col in df:
            for i, v in df[col].items():
                if(isinf(float(v)) or isnan(float(v))):
                    raise ValueError(f"Value {v} is either infinite or nan.")
                match str(i)[0]:
                    case _Prefix.BIAS.value:
                        biases.append(float(v))
                    case _Prefix.WEIGHT.value:
                        weights.append(float(v))
                    case _:
                        raise KeyError(f"Index {i} does not match any of the enumerated keys: {[e.value for e in _Prefix]}")

        return cls(len(biases), len(weights[0]), biases, weights)
    
    def copy(self) -> object:
        nodes, weights = self.shape
        return Layer(
            size=nodes, inputSize=weights, 
            biases=self.biases, weights=self.weights
        )
            
    #-------------[ OBJECT DESCRIPTORS ]-------------#

    def __len__(self) -> int:
        return len(self.biases)
    
    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.biases), len(self.weights[0]))
    
    @property
    def size(self) -> int:
        return len(self.biases) * len(self.weights[0]) + len(self.biases)
    
    def toDataFrame(self) -> DataFrame:

        data = {}
        indices = []
        nodes, weights = self.shape

        for j in range(nodes):

            if(j == 0):
                indices.append(f"{_Prefix.BIAS.value}")
            data[f"{_Prefix.NODE.value}{j}"] = [self.biases[j]]

            for k in range(weights):

                if(j == 0):
                    indices.append(f"{_Prefix.WEIGHT.value}{k}")
                data[f"{_Prefix.NODE.value}{j}"].append(self.weights[j][k])

        return DataFrame(data=data, index=indices)
    
    #-------------[ CLASS FUNCTIONS ]-------------#
                
    def calculate(self, inputs: list, activationFunction: Callable[[float, bool], float], adjustSize: bool=False):
        
        if(adjustSize):
            if(len(self) < len(inputs)):
                del self.biases[-1]
                del self.biasesGradient[-1]
                for n in range(len(self.weights)):
                    del self.weights[n][-1]
                    del self.weightsGradient[n][-1]
                    
            elif(len(self) > len(inputs)):
                self.biases.append(0.0)
                self.biasesGradient.append(0.0)
                for n in range(len(self.weights)):
                    self.weights[n].append(0.0)
                    self.weightsGradient[n].append(0.0)

        assert self.shape[1] == len(inputs)
        
        self.rawOutputs = []
        self.activatedOutputs = []

        nodes, weightsPerNode = self.shape
        for j in range(nodes):
            out = self.biases[j]
            for k in range(weightsPerNode):
                out += (self.weights[j][k] * inputs[k])
            
            self.rawOutputs.append(out) # Known as "z of L" in literature
            self.activatedOutputs.append(activationFunction(out, False)) # Known as "a of L" in literature
        
        return self.activatedOutputs.copy()
    
    def updateGradient(self, activationFunction: Callable[[float, bool], float], errors: list[float], backwardOutputs: list[float]) -> list[float]:
        """
        **Parameters**
        - *activationFunction: Callable*
        > The activation function to update the gradient with.
        - *errors: list[float]*
        > The errors calculated by the forward layer L+1 where L is this layer. 
        > For the output layer of a network, pass in the output errors resulting from the cost function.
        - *backwardOutputs: list[float]*
        > Activated outputs of the backward layer L-1.

        **Returns**
        - *backwardErrors: list[float]*
        > The backward errors for layer L-1. The hadamard product still needs to be performed on this array.
        > Used for backpropagation.
        """

        backwardErrors = []
        
        nodes, weights = self.shape
        for j in range(nodes):

            errors[j] = errors[j] * activationFunction(self.rawOutputs[j], True)
                # Performing the hadamard product node by node

            self.biasesGradient[j] += errors[j]

            for k in range(weights):
                # Weight from node j in layer L to node k in layer L-1 ==> w from k to j

                if(len(backwardErrors) < k + 1):
                    backwardErrors.append(0.0)
                backwardErrors[k] += self.weights[j][k] * errors[j]

                self.weightsGradient[j][k] += backwardOutputs[k] * errors[j]
                    # Don't need to multiply by the derived activation output of raw output z of node j since
                    # the hadamard product for j was computed earlier.

        return backwardErrors
    
    def applyGradient(self, batchSize: int, learnRate: float=1) -> None:
        
        nodes, weights = self.shape
        for j in range(nodes):
            
            self.biases[j] -= (self.biasesGradient[j] / batchSize * learnRate)
            self.biasesGradient[j] = 0.0
            
            for k in range(weights):
                
                self.weights[j][k] -= (self.weightsGradient[j][k] / batchSize * learnRate)
                self.weightsGradient[j][k] = 0.0
        