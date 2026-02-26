import torch

torch.device(0)
torch.set_default_dtype(torch.float64)
x = torch.rand([5, 3])
y = torch.rand([5, 3])
print(x)
print(y)

x.add_(y)
print(x)

# can also do z = torch.add(x, y)