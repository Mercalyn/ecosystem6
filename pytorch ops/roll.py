"""
shifts is number of shifts to do

can also do multiple shifts with shifts=(-1, 2), dims=(0, 1) and they will be respective to their dims

dim 0 = z, pos. shift goes "back"
dim 1 = y, pos. shift goes "down"
dim 2 = x, pos. shift goes "right"
"""
import torch

devc = torch.device(type="cuda")
torch.set_default_dtype(torch.float64)

i = torch.tensor(
    [[[1,2,3],
      [4,5,6]],
    
     [[11,12,13],
      [14,15,16]]],
    
    dtype=torch.float64, device=devc)

print(i)
j = torch.roll(i, shifts=1, dims=2)

print(f"\n{j}")