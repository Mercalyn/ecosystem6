"""

"""
import torch

devc = torch.device(type="cuda")
torch.set_default_dtype(torch.float64)
print("\n\n\n\n")

i = torch.tensor(
    [[[1,2,3],
    [4,5,6]],
    
    [[-11,-12,-13],
    [14,15,16]]],
    
    dtype=torch.float64, device=devc)

print(i)

hard_tanh = torch.nn.Hardtanh(-2., 2.)
k = hard_tanh(i)
print(k)
