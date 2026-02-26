"""

"""
import torch

devc = torch.device(type="cuda")
torch.set_default_dtype(torch.float64)
print("\n\n\n\n")

i = torch.tensor(
    [[[-1,0,0],
      [4,0,0],
      [0,0,7]],
    
     [[0,0,0],
      [0,5,0],
      [0,0,0]],
    
     [[0,0,0],
      [0,0,3],
      [2,0,0]]], 
    
    dtype=torch.float64, device=devc)


k = torch.transpose(i, 1, 2)
print(k)
