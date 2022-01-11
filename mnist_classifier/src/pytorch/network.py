import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

sys.path.append("../util")
from mnist_loader import load_data
from mnist_loader import mnist_data_to_list
from network_config import NetworkConfig

'''
The `MnistDigitDataset` class is inheritated from the Dataset class, where is
used to create a custom dataset. In the mnist handwritten digit dataset 
provided by Micheal Nielson, each item is composed of a vectorized gray-scale
image and the corresponding label. The most important thing we need to do is 
to retrieve the correct image and label given the `idx` in the 
`__getitem__(idx)` func.
# Reference(Creating a Custom Dataset for your file):
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
'''
class MnistDigitDataset(Dataset):
  def __init__(self, annotation_data, transform=None, target_transform=None):
    self.image_labels = annotation_data
    # self.transform = transform
    # self.target_transform = target_transform
    self.transform = None
    self.target_transform = None

  def __len__(self):
    return len(self.image_labels)

  def __getitem__(self, idx):
    image = torch.from_numpy(self.image_labels[idx][0])
    label = self.image_labels[idx][1]

    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)

    return image, label

'''
This class inheritated from the nn.Module, which defines the network
architecture. In this demo, the neural network's layer is defined as
'[784, 512, 512, 10]', which takes the `28*28` gray scale image as input, 
and the probability of 10 digits as output, and also three hidden layers with
dimension '784*512', '512*512' and '512*10' respectively.
'''
class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(784, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10),
    )

  def forward(self, x):
    # x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits


def train(dataloader, model, loss_func, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    # labels = torch.zeros(pred.shape)
    # for i in range(y.shape[0]):
    #   labels[i][y[i]] = 1
    # labels = labels.to(device)
    # loss = loss_func(pred, labels)
    loss = loss_func(pred, y)

    # Backpropagation
    # Set the gradients of all optimized torch.Tensors to zero.
    optimizer.zero_grad()
    loss.backward()
    # Perform a single optimization step for parameters updating.
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_func):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      # labels = torch.zeros(pred.shape)
      # for i in range(y.shape[0]):
      #   labels[i][y[i]] = 1
      # labels = labels.to(device)
      # test_loss += loss_func(pred, labels).item()
      test_loss += loss_func(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
  training_data, validation_data, test_data = load_data()
  training_data_list = mnist_data_to_list(training_data)
  test_data_list = mnist_data_to_list(test_data)

  training_dataset = MnistDigitDataset(training_data_list)
  test_dataset = MnistDigitDataset(test_data_list)

  training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f'Using {device} device')
  model = NeuralNetwork().to(device)
  print(model)

  config = NetworkConfig(batch_size=50, learning_rate=0.8, \
                         activation_func_type="Sigmoid", \
                         loss_func_type = "CrossEntropyLoss")
                        #  loss_func_type = "MeanSquareLoss")
  '''
  Creating loss/cost function.
  #Ref 'https://pytorch.org/docs/stable/nn.html#loss-functions' for 
  more loss functions.
  '''
  if config.loss_func_type == 'MeanSquareLoss':
    loss_func = nn.MSELoss(reduction='mean')
  elif config.loss_func_type == 'CrossEntropyLoss':
    loss_func = nn.CrossEntropyLoss()

  # See 'https://pytorch.org/docs/stable/optim.html'.
  # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
  optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

  epochs = 10
  for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------------")
    train(training_dataloader, model, loss_func, optimizer)
    test(test_dataloader, model, loss_func)