import torch

devc = torch.device(type="cuda")
torch.set_default_dtype(torch.float64)

"""
exact quantile
>>> 2
"""
j = torch.tensor(
    [
        [[1,2,3],
        [4,5,6],
        [7,8,9]],
    
        [[11,12,13],
        [14,15,16],
        [17,18,19]],
        
        [[99,99,99],
        [99,99,99],
        [99,99,99]],
    ],
    
    dtype=torch.float64, device=devc)

k = torch.quantile(j, q=.5, dim=1, keepdim=True)
print(k)

