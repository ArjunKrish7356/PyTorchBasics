""" 
Dataset: CIFAR-10 — 60,000 32×32 RGB images across 10 classes (e.g., airplane, car, bird, etc.).

Model Architecture: This is a Convolutional Neural Network (CNN).
  Convolutional Layer (3 input channels, 32 output channels, 3x3 kernel)
  ReLU activation
  Max Pooling Layer (3x3 kernel, stride 2)
  Flattens the output of the convolutional layers into vectors
  Linear layer (input to 512 units)
  ReLU activation
  Linear layer (512 to 10 units, one per class)

Optimizer: AdamW
Loss Function: CrossEntropyLoss

Training: 5 epochs
Test Accuracy: ~65.3%
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Downloading the training dataset
training_data = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

# Downloading the testing dataset
test_data = datasets.CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# Class are the labels of the things we want to predict. There are 10 labels in CIFAR10
# {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
classes = training_data.classes
class_map = {}
for i,item in enumerate(classes):
    class_map[i] = item
print(class_map)

batch_size = 64

# Wrapping dataset inside a DataLoader object, the dataset will be batches inside the dataloader with each batch having 64 items.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# auto selecting device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Defining the Neural network and adding the required layers.
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_stack = nn.Sequential(
        nn.Conv2d(3,32,3),
        nn.ReLU(),
        nn.MaxPool2d(3,2),
    )
    self.linear_relu_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*14*14, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )

  def forward(self, x):
    x = self.conv_stack(x)
    logits = self.linear_relu_stack(x)
    return logits

# Creating a class of NeuralNetwork, Optimizer and Loss Function.
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

# Training logic
def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    pred = model(X)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      loss, current = loss.item(), (batch + 1)* len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Testing logic
def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Training the model epoch number of times.
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
