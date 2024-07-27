import math
import random
import threading

import Functions

class Layer:
    
    UPPER_THRESHOLD = 1.0
    LOWER_THRESHOLD = -1.0
    
    def __init__(
        self, manifestFile: str, layerSize: int, numInputs: int,
        biases = None, biasesGradient = None,
        weights = None, weightsGradient = None,
        zVector = None, outputVector = None
    ):
        
        self.manifestFile = manifestFile
        self.layerSize = layerSize
        self.numInputs = numInputs
        
        if(biases is None): self.biases = [] 
        else: self.biases = biases
        
        if(biasesGradient is None): self.biasesGradient = []
        else: self.biasesGradient = biasesGradient
        
        if(weights is None): self.weights = []
        else: self.weights = weights
        
        if(weightsGradient is None): self.weightsGradient = []
        else: self.weightsGradient = weightsGradient
        
        if(zVector is None): self.zVector = []
        else: self.zVector = zVector
        
        if(outputVector is None): self.outputVector = []
        else: self.outputVector = outputVector
     
        for n in range(self.layerSize):
            
            if(biases is None): self.biases.append(0.0)
            if(biasesGradient is None): self.biasesGradient.append(0.0)
            if(weights is None): self.weights.append([])
            if(weightsGradient is None): self.weightsGradient.append([])
            
            for w in range(self.numInputs):
                
                if(weights is None): self.weights[n].append(random.uniform(self.LOWER_THRESHOLD, self.UPPER_THRESHOLD))
                if(weightsGradient is None): self.weightsGradient[n].append(0.0)
                
    def calculate(self, inputs: list):
        
        assert self.numInputs == len(inputs)
        
        outputVector = []
        zVector = []
        
        for n in range(self.layerSize):         
            # out = 0
            out = self.biases[n]
            
            for i in range(self.numInputs):
                out += (self.weights[n][i] * inputs[i])
            
            zVector.append(out)
            outputVector.append(Functions.bipolarSigmoid(out))
        
        self.outputVector = outputVector.copy()
        self.zVector = zVector.copy()
        
        return outputVector
    
    def updateGradient(self, errors: list, previousActivations: list):
        
        assert self.layerSize == len(errors)
        assert self.numInputs == len(previousActivations)
        
        for n in range(self.layerSize):
            
            self.biasesGradient[n] += (errors[n])
            
            for w in range(self.numInputs):
                self.weightsGradient[n][w] += (previousActivations[w] * errors[n])
    
    def applyGradient(self, miniBatchSize: int, learnRate: float):
        
        for n in range(self.layerSize):
            
            d = self.biasesGradient[n] / miniBatchSize * learnRate
            self.biases[n] -= d
            self.biasesGradient[n] = 0
            
            for w in range(self.numInputs):
                
                d = self.weightsGradient[n][w] / miniBatchSize * learnRate
                self.weights[n][w] -= d
                self.weightsGradient[n][w] = 0
                    
    def regularization(d: float, current: float):
        
        return (d * (1.0 / abs(current)))

    def save(self):
                
        lines = []
        
        for n in range(self.layerSize):
            
            lines.append(str(self.biases[n]) + ";")

            for w in range(self.numInputs):
                
                if(w == self.numInputs - 1):
                    lines[n] += str(self.weights[n][w])
                else:
                    lines[n] += (str(self.weights[n][w]) + ",")
        
        with open(self.manifestFile, "w") as file:
            for l in lines:
                file.write(l + "\n")
            file.close()
        