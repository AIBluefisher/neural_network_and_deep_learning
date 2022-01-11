import torch
from torch import nn

def init_normal(m):
  if type(m) == nn.Linear:
    nn.init.normal_(m.weight, mean=0, std=0.01)
    nn.init.zeros_(m.bias)

def init_constant(m):
  if type(m) == nn.Linear:
    nn.init.constant_(m.weight, 1)
    nn.init.zeros_(m.bias)

net = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))
net.apply(init_normal)
print("normal weight: {0}, \nnormal bias: {1}".format(net[0].weight.data[0], net[0].bias.data[0]))
print("normal weight: {0}, \nnormal bias: {1}".format(net[0].weight.data, net[0].bias.data))

net.apply(init_constant)
print("constant weight: {0}, constant bias: {1}".format(net[0].weight.data[0], net[0].bias.data[0]))
