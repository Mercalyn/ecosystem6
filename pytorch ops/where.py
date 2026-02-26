import torch

devc = torch.device(type="cuda")
torch.set_default_dtype(torch.float64)

i = torch.tensor(
    [[[1,2,3],
    [4,5,6]],
    
    [[11,12,13],
    [14,15,16]]],
    
    dtype=torch.float64, device=devc)


"""
element-wise, one-to-one
"""
j = torch.tensor(
    [[[0,0,0],
    [1,1,1]],
    
    [[1,2,0],
    [0,0,0]]],
    
    dtype=torch.float64, device=devc)

k = torch.where(j > 0, i, 0) # where condition, assign (number, tensor) if true, assign if false
# j here is acting as a mask

#print(k)


"""
x dim broadcast
"""
j = torch.tensor(
    [1,0,1],
    dtype=torch.float64, device=devc)

k = torch.where(j > 0, i, 0)
#print(k)


"""
y dim broadcast
"""
l = torch.tensor(
    [[1], # [1,0] does not work luckily, no room for misjudging dims
     [0]],
    dtype=torch.float64, device=devc)

m = torch.where(l > 0, i, 0)
#print(m)


"""
z dim broadcast
"""
o = torch.tensor(
    [[[1]],
     [[0]]],
    dtype=torch.float64, device=devc)

p = torch.where(o > 0, i, 0)
print(p)