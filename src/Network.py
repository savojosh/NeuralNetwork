import os
import MathFunctions
from pandas import read_csv
from DataPoint import DataPoint
from Layer import Layer
from Tools import *
from typing import Callable
from shutil import rmtree

class Network:

    #-------------[ CONSTRUCTORS ]-------------#

    def __init__(self, inputSize: int, layerSizes: list[int]|tuple[int], layers: list[Layer]=None, lower: float=0, upper: float=1) -> object:

        if(layers is None):
            self.layers = []
            inputs = inputSize
            for ls in layerSizes:
                self.layers.append(Layer(
                    size=ls, inputSize=inputs, lower=lower, upper=upper
                ))
                inputs = ls
        else:
            self.layers = layers

    @classmethod
    def fromFolder(cls, folder: str):
        layers = []
        for p in os.listdir(folder):
            path = buildPath(folder, p)
            if(os.path.isfile(path)):
                layers.append(Layer.fromDataFrame(read_csv(path, index_col=0)))

        assert len(layers) > 0

        return cls(0, 0, layers)

    def copy(self) -> object:
        copiedLayers = [l.copy() for l in self.layers]
        return Network(layers=copiedLayers)

    #-------------[ OBJECT DESCRIPTORS ]-------------#

    def __len__(self) -> int:
        return len(self.layers)
    
    @property
    def shape(self) -> tuple:
        return tuple(len(l) for l in self.layers)
    
    @property
    def size(self) -> int:
        return sum(l.size for l in self.layers)
    
    @property
    def inputSize(self) -> int:
        return self.layers[0].shape[1]
    
    #-------------[ CLASS FUNCTIONS ]-------------#
        
    def calculate(self, dp: DataPoint, activationFunction: Callable[[float, bool], float], adjustSize: bool=False):
        
        if(not adjustSize):
            assert self.layers[0].shape[1] == dp.xSize
            assert self.layers[len(self.layers) - 1].shape[0] == dp.ySize
        
        outputs = self.layers[0].calculate(dp.values, activationFunction, adjustSize)
        
        for l in range(1, len(self.layers)):
            
            outputs = self.layers[l].calculate(outputs, activationFunction, adjustSize)

        return outputs
    
    def learn(self, data: list[DataPoint], activationFunction: Callable[[float, bool], float], learnRate: float, adjustSize: bool=False):

        cost = 0
        expectedDistribution = [0 for i in range(data[0].ySize)]
        chosenDistribution = [0 for i in range(data[0].ySize)]
        correctDistribution = [0 for i in range(data[0].ySize)]

        for d in data:

            outputs = self.calculate(d, activationFunction, adjustSize)
            errors = [MathFunctions.mse(a, y, derivative=True) for a, y in zip(outputs, d.y)]

            for l in reversed(range(len(self.layers))):
                backwardOutputs = self.layers[l - 1].activatedOutputs if l > 0 else d.values
                errors = self.layers[l].updateGradient(activationFunction, errors, backwardOutputs)

            cost += MathFunctions.average([MathFunctions.mse(a, y, derivative=False) for a, y in zip(outputs, d.y)])

            expectedDistribution[d.label] += 1
            choice = max(range(len(outputs)), key=outputs.__getitem__)
            chosenDistribution[choice] += 1
            correctDistribution[choice] += (1 if d.label == choice else 0)

        for l in self.layers:
            l.applyGradient(len(data), learnRate)

        cost /= len(data)

        return (cost, sum(correctDistribution) / len(data), expectedDistribution, chosenDistribution, correctDistribution)
    
    def save(self, folder: str):
        
        rmtree(folder, ignore_errors=True) 
        os.makedirs(folder, exist_ok=True)
        
        for l in range(len(self.layers)):
            
            self.layers[l].toDataFrame().to_csv(buildPath(folder, f"Layer_{l:02}.csv"))
                        