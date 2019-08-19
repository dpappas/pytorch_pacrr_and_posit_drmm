
import torch

# emb_size    = 100
# ngram       = 3
# input       = torch.autograd.Variable(torch.randn(20, 1, 50, emb_size))
# print(input.size())
# m           = torch.nn.Conv2d(1, emb_size, kernel_size=(ngram, emb_size), stride=(1, 1), padding=(ngram-1, 0))
# output      = m(input).transpose(1, 3)[:,:,:input.size(-2),:]
# print(m.weight.size())
# print(output.size())
#
# print(20 * '-')
#
# emb_size    = 100
# input       = input.squeeze(1).transpose(-1,-2)
# print(input.size())
# m           = torch.nn.Conv1d(emb_size, emb_size, ngram, padding=ngram-1, bias=True)
# output      = m(input)[:,:,:input.size(-1)]
# print(m.weight.size())
# print(output.size())



class LINEAR(torch.nn.Module):
    def __init__(self):
        super(LINEAR, self).__init__()
        # self.w  = torch.autograd.Variable(torch.zeros(2, 1), requires_grad=True)
        # self.w  = torch.nn.Parameter(torch.zeros(2, 1))
        # torch.nn.init.xavier_uniform_(self.w, gain=1)
        self.w  = torch.nn.Parameter(torch.randn(2, 1).uniform_(-1e-4, 1e-4))
    def forward(self, x):
        x       = torch.autograd.Variable(torch.FloatTensor(x), requires_grad=False)
        return torch.mm(x, self.w)

model       = LINEAR()
params      = list(model.parameters()) #+ [model.w]
print(params)
# exit()
optimizer   = torch.optim.SGD(params, lr=0.01)

model.train()
for i in  range(50):
    my_x        = [[1,2]]
    y           = model(my_x)
    cost_       = torch.abs(1 - y)
    print(y, cost_)
    cost_.backward()
    optimizer.step()
    optimizer.zero_grad()


def my_cosine_sim(in1, in2):
    ret = []
    for i in range(in1.size(1)):
        simi = F.cosine_similarity(in1[:,i,:].expand_as(in2), in2, dim=-1, eps=1e-8)
        ret.append(simi)
    ret = torch.stack(ret, dim=1)
    return ret

def my_cosine_sim2(A, B):
        A_mag       = torch.norm(A, 2, dim=2)
        B_mag       = torch.norm(B, 2, dim=2)
        num         = torch.bmm(A, B.transpose(-1, -2))
        den         = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
        print(den)
        dist_mat    = num / den
        return dist_mat

import torch.nn.functional as F
# in1     = torch.randn(1, 4, 30)
# in2     = torch.randn(1, 9, 30)
in1     = torch.zeros(1, 4, 30) + 1e-8
in2     = torch.randn(1, 9, 30) + 1e-8
simi    = my_cosine_sim2(in1, in2)
print(simi.size())
print(simi)



