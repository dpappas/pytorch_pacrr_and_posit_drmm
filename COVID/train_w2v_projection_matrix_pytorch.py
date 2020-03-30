
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from pprint import pprint
import numpy as np
import torch.autograd as autograd

w2v_bin_path_old    = '/home/dpappas/COVID/COVID/pubmed2018_w2v_30D.bin'
w2v_bin_path_new    = '/home/dpappas/COVID/covid_19_w2v_embeds_30.model'
wv_old              = KeyedVectors.load_word2vec_format(w2v_bin_path_old, binary=True)
wv_new              = Word2Vec.load(w2v_bin_path_new)

common_tokens       = sorted(list(set(wv_old.vocab.keys()).intersection(set(wv_new.wv.vocab.keys()))))

A_matrix            = np.stack([wv_new[tok] for tok in common_tokens], axis=0)
B_matrix            = np.stack([wv_old[tok] for tok in common_tokens], axis=0)

print(A_matrix.shape)
print(B_matrix.shape)

import torch
import torch.optim as optim
A_matrix            = autograd.Variable(torch.FloatTensor(A_matrix), requires_grad=False)
B_matrix            = autograd.Variable(torch.FloatTensor(B_matrix), requires_grad=False)

print(A_matrix.size())
print(B_matrix.size())

X1                  = torch.nn.Parameter(torch.randn(30,30))
# optimizer           = optim.Adam([X1], lr=1.000)
optimizer           = optim.RMSprop([X1], lr=0.001, alpha=0.99, eps=1e-08, momentum=0.1, centered=False)

for i in range(1000):
    Y               = torch.mm(A_matrix, X1)
    cost            = torch.sum(torch.norm(Y - B_matrix, p=2, dim=1)) # + torch.sum(torch.norm(Y - B_matrix, p=1, dim=1))
    print(cost)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()

q = autograd.Variable(torch.FloatTensor([wv_new['the']]), requires_grad=False)
r = torch.mm(q, X1).squeeze().detach().numpy()

pprint(wv_old.similar_by_vector(r))

# import  torch.nn                as nn
# import  torch.nn.functional     as F
#
# l1          = nn.Linear(30, 120, bias=True)
# l2          = nn.Linear(120, 30, bias=True)
# # optimizer   = optim.Adam(l1.parameters()+l2.parameters(), lr=0.01)
# optimizer   = optim.RMSprop(l1.parameters()+l2.parameters(), lr=0.01)
#
# for i in range(1000):
#     Y               = l2(F.sigmoid(l1(A_matrix)))
#     cost            = torch.sum(torch.norm(Y - B_matrix, p=2, dim=1))
#     print(cost)
#     cost.backward()
#     optimizer.step()
#     optimizer.zero_grad()


