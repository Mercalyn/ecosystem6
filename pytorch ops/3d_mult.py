"""
this works, and will be the basis of my evolution matrix multiplication.
a z index(depth) basically ignores other z indices and 2d matrix multiplies by its corresponding depth's 2d matrix. 
this means that x[0] which is [[1,2,3],[4,5,6]] is mat mult with [[7,8],[9,10],[11,12]] and is independent of x[1]. 
--
i would make the depth axis the population, and have seperate tensors listed for neural layer depth
--
bias will element-wise add its corresponding layers' bias
--
squash func im thinking will 
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

j = torch.tensor(
    [[[7,8],
      [9,10],
      [11,12]],
    
     [[17,18],
      [19,20],
      [21,22]]],
    
    dtype=torch.float64, device=devc)

k = torch.matmul(i, j)
print(k)