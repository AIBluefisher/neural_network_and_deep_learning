import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2,4))
net(X)

print(net[2].state_dict())

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

net[2].weight.grad = None

print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])