
import torch
sz=34
tmp=(torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
# tmp=(torch.triu(torch.ones(sz, sz)) == 1)
print(tmp.numpy())