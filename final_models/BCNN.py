
import  os
import  json
import  time
import  random
import  logging
import  subprocess
import  torch
import  torch.nn.functional         as F
import  torch.nn                    as nn
import  numpy                       as np
import  torch.optim                 as optim
# import  cPickle                     as pickle
import  pickle
import  torch.autograd              as autograd
from    tqdm                        import tqdm
from    pprint                      import pprint
from    gensim.models.keyedvectors  import KeyedVectors
from    nltk.tokenize               import sent_tokenize
from    difflib                     import SequenceMatcher
import  re







class BCNN(nn.Module):
    def __init__(self, embedding_dim=30, additional_feats=8, convolution_size=4):
        super(BCNN, self).__init__()
        self.additional_feats   = additional_feats
        self.convolution_size   = convolution_size
        self.embedding_dim      = embedding_dim
        self.conv1              = nn.Conv1d(
            in_channels         = self.embedding_dim,
            out_channels        = self.embedding_dim,
            kernel_size         = self.convolution_size,
            padding             = self.convolution_size-1,
            bias                = True
        )
        self.linear_out         = nn.Linear(self.additional_feats+3, 2, bias=True)
        self.conv1_activ        = torch.nn.Tanh()
    def my_cosine_sim(self, A, B):
        # A     = A.unsqueeze(0)
        # B     = B.unsqueeze(0)
        A_mag = torch.norm(A, 2, dim=2)
        B_mag = torch.norm(B, 2, dim=2)
        num = torch.bmm(A, B.transpose(-1, -2))
        den = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
        dist_mat = num / den
        return dist_mat
    def apply_one_conv(self, batch_x1, batch_x2):
        batch_x1_conv       = self.conv1(batch_x1)
        batch_x2_conv       = self.conv1(batch_x2)
        #
        x1_window_pool      = F.avg_pool1d(batch_x1_conv, self.convolution_size, stride=1)
        x2_window_pool      = F.avg_pool1d(batch_x2_conv, self.convolution_size, stride=1)
        #
        x1_global_pool      = F.avg_pool1d(batch_x1_conv, batch_x1_conv.size(-1), stride=None)
        x2_global_pool      = F.avg_pool1d(batch_x2_conv, batch_x2_conv.size(-1), stride=None)
        #
        sim                 = self.my_cosine_sim(x1_global_pool.transpose(1,2), x2_global_pool.transpose(1,2))
        sim                 = sim.squeeze(-1).squeeze(-1)
        return x1_window_pool, x2_window_pool, x1_global_pool, x2_global_pool, sim
    def forward(self, batch_x1, batch_x2, batch_y, batch_features):
        batch_x1        = autograd.Variable(torch.FloatTensor(batch_x1),        requires_grad=False)
        batch_x2        = autograd.Variable(torch.FloatTensor(batch_x2),        requires_grad=False)
        batch_y         = autograd.Variable(torch.LongTensor(batch_y),          requires_grad=False)
        batch_features  = autograd.Variable(torch.FloatTensor(batch_features),  requires_grad=False)
        if(use_cuda):
            batch_x1        = batch_x1.cuda()
            batch_x2        = batch_x2.cuda()
            batch_y         = batch_y.cuda()
            batch_features  = batch_features.cuda()
        #
        x1_global_pool      = F.avg_pool1d(batch_x1.transpose(-1,-2), batch_x1.size(-1), stride=None)
        x2_global_pool      = F.avg_pool1d(batch_x2.transpose(-1,-2), batch_x1.size(-1), stride=None)
        sim1                = self.my_cosine_sim(x1_global_pool.transpose(1,2), x2_global_pool.transpose(1,2))
        sim1                = sim1.squeeze(-1).squeeze(-1)
        #
        (x1_window_pool, x2_window_pool, x1_global_pool, x2_global_pool, sim2) = self.apply_one_conv(batch_x1.transpose(1,2), batch_x2.transpose(1,2))
        (x1_window_pool, x2_window_pool, x1_global_pool, x2_global_pool, sim3) = self.apply_one_conv(x1_window_pool, x2_window_pool)
        #
        mlp_in              = torch.cat([sim1.unsqueeze(-1), sim2.unsqueeze(-1), sim3.unsqueeze(-1), batch_features], dim=-1)
        mlp_out             = self.linear_out(mlp_in)
        mlp_out             = F.softmax(mlp_out, dim=-1)
        #
        cost                = F.cross_entropy(mlp_out, batch_y, weight=None, reduction='elementwise_mean')
        #
        return cost



use_cuda = False

embedding_dim       = 30
additional_feats    = 8
b_size              = 200
max_len             = 40
lr                  = 0.08

model = BCNN(
    embedding_dim       = embedding_dim,
    additional_feats    = additional_feats,
    convolution_size    = 4
)
model.train()

params      = model.parameters()
# optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0004)
optimizer   = optim.Adagrad(params, lr=lr, lr_decay=0, weight_decay=0.0004, initial_accumulator_value=0)

bx1         = np.random.randn(b_size, max_len, embedding_dim)
bx2         = np.random.randn(b_size, max_len, embedding_dim)
by          = np.random.randint(2, size=b_size)
bf          = np.random.randn(b_size, 8)

for i in range(100):
    cost_ = model(
        batch_x1        = bx1,
        batch_x2        = bx2,
        batch_y         = by,
        batch_features  = bf
    )
    cost_.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(cost_)



























