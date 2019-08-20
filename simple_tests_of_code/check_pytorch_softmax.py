

import torch
import  torch.nn.functional         as F

d = torch.FloatTensor([[0.88, 1.5454]])
print(d.size())
print(d)

print(F.softmax(d, dim=-1))






