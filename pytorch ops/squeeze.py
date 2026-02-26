import torch

devc = torch.device(type="cuda")
torch.set_default_dtype(torch.float64)

i = torch.zeros(5, 1, 5)
print(f"{i.size()}")

j = torch.squeeze(i, dim=1)
print(f"{j.size()}")