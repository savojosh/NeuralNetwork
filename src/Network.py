import os
import Functions
from DataPoint import DataPoint
from Layer import Layer

class Network:
    
    def __init__(self, manifestFolder: str, numInputs: int, layerSizes: list=None):
        
        self.manifestFolder = manifestFolder.replace("\\", "/")
        self.numInputs = numInputs
        self.layers = []
        self.performance = 0.0
        self.cost = 0.0
        
        if(os.path.isdir(self.manifestFolder)):
            
            files = os.listdir(self.manifestFolder)
            files = [(self.manifestFolder + "/" + f) for f in files if os.path.isfile(self.manifestFolder + '/' + f)]
            
            for f in files:
                
                with open(f, "r") as file:
                                        
                    lines = file.readlines()
                    
                    sBiases = []
                    sWeights = []
                    
                    for l in lines:
                        
                        l = l.strip()
                        sBiases.append(float(l.split(";")[0]))
                        sWeights.append([float(w) for w in l.split(";")[1].split(",")])
                        
                    self.layers.append(Layer(
                        file.name.replace("\\", "/"),
                        len(sBiases),
                        len(sWeights[0]),
                        biases=sBiases,
                        weights=sWeights
                    ))
                    
                    file.close()  
        
        else:
                    
            self.layers.append(Layer(
                self.manifestFolder + "/Layer_" + f'{1:02d}' + ".txt",
                layerSizes[0],
                self.numInputs
            ))
            
            for l in range(1, len(layerSizes)):
                
                self.layers.append(Layer(
                    manifestFile = self.manifestFolder + "/Layer_" + f"{l + 1:02d}" + ".txt",
                    layerSize = layerSizes[l],
                    numInputs = layerSizes[l - 1]
                ))
        
        self.networkSize = len(self.layers)
        
    def calculate(self, dp: DataPoint):
        
        assert self.numInputs == len(dp.values)
        assert self.layers[len(self.layers) - 1].layerSize == len(dp.y)
        
        outputs = self.layers[0].calculate(dp.values)
        
        for l in range(1, self.networkSize):
            
            outputs = self.layers[l].calculate(outputs)

        return outputs
    
    def learn(self, data: list, learnRate: float):
        
        outputs = []
        cost = 0.0
        
        for dp in range(len(data)):
            
            # if(dp != 0 and dp % int(len(data) / 3) == 0): 
            #     # print(f'{dp:05d}')
            #     self.cost = cost / dp
            #     print("  " + f'{self.cost:.5f}')

            outputs.append(self.calculate(data[dp]))
            errors = []
            
            for o in range(len(data[dp].y)):
                
                a = outputs[dp][o]
                y = data[dp].y[o]
                z = self.layers[len(self.layers) - 1].zVector[o]
                
                errors.append((2 * (a - y)) * (Functions.dBipolarSigmoid(z)))
                
                cost += pow(a - y, 2)
                
            if(len(self.layers) > 1):
                
                self.layers[len(self.layers) - 1].updateGradient(errors, self.layers[len(self.layers) - 2].outputVector)

                for l in range(len(self.layers) - 2, -1, -1):
                    
                    previousErrors = errors.copy()
                    outWeights = self.layers[l + 1].weights
                    errors = []
                    
                    for cn in range(self.layers[l].layerSize):
                        
                        errors.append(0.0)
                        
                        for nn in range(len(outWeights)):
                            
                            w = outWeights[nn][cn]
                            e = previousErrors[nn]
                            z = self.layers[l].zVector[cn]
                            
                            errors[cn] += w * e * Functions.dBipolarSigmoid(z)
                        
                        errors[cn] = errors[cn] / len(outWeights)
                        
                    if(l > 0):
                        
                        self.layers[l].updateGradient(errors, self.layers[l - 1].outputVector)
                        
                    else:
                        
                        self.layers[l].updateGradient(errors, data[dp].values)
                    
            else:
                
                self.layers[len(self.layers) - 1].updateGradient(errors, data[dp].values)
            
        for l in self.layers:
            
            l.applyGradient(len(data), learnRate)
                
        self.cost = cost / len(data)
        
        return outputs
    
    def save(self):
        
        os.makedirs(self.manifestFolder, exist_ok=True)
        
        for l in range(self.networkSize):
            
            self.layers[l].save()
            
    def setManifestFolder(self, manifestFolder: str):
        
        self.manifestFolder = manifestFolder.replace("\\", "/")
        
        for l in range(len(self.layers)):
            
            self.layers[l].manifestFile = self.manifestFolder + "/Layer_" + f'{l:02d}' + ".txt"
