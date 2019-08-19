
import torch
A           = torch.rand(10,5)
B           = torch.rand(10,5)
A           = A.unsqueeze(0)
B           = B.unsqueeze(0)
A_mag       = torch.norm(A, 2, dim=2)
B_mag       = torch.norm(B, 2, dim=2)
num         = torch.bmm(A, B.transpose(-1, -2))
den         = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
dist_mat    = num / den
print(dist_mat)
print(dist_mat.size())

import numpy as np

# from scipy.spatial.distance import cosine
# print cosine(A.data.numpy()[0][0], B.data.numpy()[0][0])
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(A.data.numpy()[0], B.data.numpy()[0]))

