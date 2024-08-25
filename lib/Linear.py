import numpy as np

class linear_layer:
    def __init__(self,infeacher, outfeacher):
        super(linear_layer, self).__init__()
        self.W = np.random.rand(outfeacher, infeacher) - 0.5
        self.B = np.random.rand(outfeacher,1) -0.5
    
    def forward(self, x):
        self.out = self.W.dot(x) + self.B
        # Z1 = W1.dot(X_dev) + b1
        return self.out


# class linear_layer:
#     def __init__(self, infeacher, outfeacher):
#         self.W = np.random.randn(outfeacher, infeacher) * np.sqrt(2.0 / (infeacher + outfeacher))
#         self.B = np.zeros((outfeacher, 1))

#     def forward(self, x):
#         self.out = self.W.dot(x) + self.B
#         return self.out
