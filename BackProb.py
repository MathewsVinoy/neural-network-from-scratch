import numpy as np

# def one_hot(Y):
#     one_hot_y = np.zeros((Y.size, Y.max()+1))
#     one_hot_y[np.arange(Y.size, Y)]=1
#     one_hot_y=one_hot_y.T
#     return one_hot_y

class BackProb_all:
    def __init__(self):
        pass

    def one_hot(self, Y):
        one_hot_y = np.zeros((Y.size, 10))
        one_hot_y[np.arange(Y.size), Y] = 1
        return one_hot_y
    
    def Relu_deriv(self,z):
        return z > 0
    
    def BackProb(self,Y,output,out4,w2,w3,out3,X,m):
        one_hot_y = self.one_hot(Y)
        dZ3 = output - one_hot_y.T
        dW3 = 1/m * dZ3.dot(out4.T)
        db3 = 1/m * np.sum(dZ3)
        dZ2 = w3.T.dot(dZ3)* self.Relu_deriv(out3)
        dW2 = 1/m * dZ2.dot(X.T)
        db2 = 1/m * np.sum(dZ2)
        # dZ1 = w2.T.dot(dZ2)* self.Relu_deriv(out1)
        # dW1 = 1/m * dZ1.dot(X.T)
        # db1 = 1/m * np.sum(dZ1)
        return dW3, db3,dW2, db2, #dW1, db1