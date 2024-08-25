import pandas as pd
from Model import Model
from lib.cCrossEntropy_loss import CrossEntropyLoss
from BackProb import BackProb_all
from lib.batches import get_batch
from update_value import UpdateParameter
from lib.accuracy import get_accuraacy


train_data = pd.read_csv('data/mnist_train.csv')
test_data = pd.read_csv('data/mnist_test.csv')

train_batches, M= get_batch(data=train_data, batch_size=32)
test_batches, _ = get_batch(data=test_data, batch_size=32)

model = Model()
bp = BackProb_all()

EPOCHS =500

for epoch in range(EPOCHS):
    #training part
    acc=0
    for batch in train_batches:
        x, y = batch
        out = model.forward(X=x)
        dW3, db3,dW2, db2 = bp.BackProb(
            # out1=model.out1,
            # out2=model.out2,
            out3=model.out3,
            out4=model.out4,
            output=out,
            Y=y,
            w2=model.W2,
            w3=model.W3,
            X=x,
            m=M
        )
        UpdateParameter(
            lr=0.10,
        
            db2=db2,
            db3=db3,
            
            dW2=dW2,
            dW3=dW3,
            model=model
        )
        acc += get_accuraacy(out,y)

    print(f'Epoch {epoch+1}, train accuracy:{acc/len(train_batches)}')

