import numpy as np

class Relu:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)
        return self.output