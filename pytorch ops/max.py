import torch

devc = torch.device(type="cuda")
torch.set_default_dtype(torch.float64)

"""
reduction to find max
"""

# find highest in flattened, no dim named
i = torch.tensor(
    [[[1,2,3],
      [4,5,6]],
     
     [[4,8,9],
      [0,0,0]],
    
    [[-11,-12,-13],
     [ 14, 15, 16]]],
    
    dtype=torch.float64, device=devc)

k = torch.max(i) # 16
print(k)


# dim named, returns values as max values, indices as the this dimension index
l = torch.max(i, dim=2) # 
print(l)