import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True,
)

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True,
)
test_dataloader = DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=False,
)

output = [1,2,3,4,5,6,7,8,9,0]

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=784,out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.seq(x)
        return x
    
model = MnistModel()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

from tqdm.auto import tqdm

epochs =5
for epoch in range(epochs):
    train_loss =0
    test_loss =0
    model.train()
    for batch in tqdm(train_dataloader):
        inputs, labels = batch
        inputs = inputs.view(-1, 784)
        output = model(inputs)
        loss = loss_fn(output, labels)
        train_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            inputs, labels = batch
            inputs = inputs.view(-1,784)
            output = model(inputs)
            loss = loss_fn(output, labels)
            test_loss = loss.item()
    
    print(f"train loss {train_loss}, test loss {test_loss}")

from pathlib import Path
MODEL_PATH = Path("models")
MODEL_NAME = "model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Create the directory if it doesn't exist
MODEL_PATH.mkdir(parents=True, exist_ok=True)
#Save
torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)