import torch
from torch import log2, tensor
X = tensor(0.75, dtype=torch.float32)
res = -log2(X)
print(res)
