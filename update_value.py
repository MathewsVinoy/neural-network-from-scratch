
def UpdateParameter(lr,dW3, db3,dW2, db2,model):
    # model.W1 -= lr*dW1
    # model.B1 -= lr*db1
    model.W2 -= lr*dW2
    model.B2 -= lr*db2
    model.W3 -= lr*dW3
    model.B3 -= lr*db3