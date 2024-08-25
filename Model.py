from lib.Linear import linear_layer
from lib.Relu import Relu
from lib.softmax import Softmax

class Model:
    def __init__(self):
        # self.linear1 = linear_layer(infeacher=784,outfeacher=10)
        # self.W1 = self.linear1.W
        # self.B1 = self.linear1.B
        self.Relu = Relu()
        self.linear2 = linear_layer(infeacher=784,outfeacher=10)
        self.W2 = self.linear2.W
        self.B2 = self.linear2.B
        self.linear3 = linear_layer(infeacher=10,outfeacher=10)
        self.W3 = self.linear3.W
        self.B3 = self.linear3.B

    def forward(self,X):
        # self.out1= self.linear1.forward(x=X)
        # self.out2 = self.Relu.forward(self.out1)
        self.out3 = self.linear2.forward(x=X)#self.out2
        self.out4 = self.Relu.forward(self.out3)
        self.out5 = self.linear3.forward(x=self.out4)
        self.output = Softmax(inputs=self.out5)
        return self.output