"""
Dataset: CIFAR-10 — 60,000 32×32 RGB images across 10 classes (e.g., airplane, car, bird, etc.).

Model Architecture: This is a deeper Convolutional Neural Network (CNN).
  Conv2D Layer (3 input channels, 32 output channels, 3x3 kernel)
  BatchNorm, ReLU, MaxPooling
  Conv2D Layer (32 → 64), BatchNorm, ReLU, MaxPooling
  Conv2D Layer (64 → 128), BatchNorm, ReLU, AdaptiveMaxPool(4x4)
  Flatten
  Linear Layer (2048 → 512)
  ReLU + Dropout
  Linear Layer (512 → 10 classes)

Optimizer: AdamW  
Loss Function: CrossEntropyLoss  
Learning Rate Scheduler: StepLR (step size = 7)

Training: 30 epochs  
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

# Visualizing a few training samples
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(class_map[label])
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0))
plt.show()

batch_size = 128

# Wrapping dataset inside a DataLoader object, the dataset will be batched inside the dataloader with each batch having 128 items.
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
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(32,64,3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(64,128,3),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveMaxPool2d((4,4))
    )
    self.linear_relu_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128*4*4, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 10)
    )

  def forward(self, x):
    x = self.conv_stack(x)
    logits = self.linear_relu_stack(x)
    return logits

# Creating a class of NeuralNetwork, Optimizer, Loss Function and Learning Rate Scheduler
model = NeuralNetwork().to(device)
print(device)

from torch.optim.lr_scheduler import StepLR

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
Scheduler = StepLR(optimizer, 7)

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

# Training the model for 30 epochs
epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    Scheduler.step()
print("Done!")
