
import json, torch, re, pickle, random, os, sys
import  torch.nn as nn
import numpy as np
from torch import FloatTensor as FT
from tqdm import tqdm
from torch.optim import Adam

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

######################################################################

use_cuda    = torch.cuda.is_available()
device      = torch.device("cuda") if(use_cuda) else torch.device("cpu")

######################################################################

class SGNS(nn.Module):
    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding  = embedding
        self.vocab_size = vocab_size
        self.n_negs     = n_negs
        self.weights    = None
        if weights is not None:
            wf              = np.power(weights, 0.75)
            wf              = wf / wf.sum()
            self.weights    = FT(wf)
    def forward(self, true_vecs, out_vecs):
        batch_size      = true_vecs.size()[0]
        context_size    = true_vecs.size()[1]
        if self.weights is not None:
            nwords  = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords  = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long().to(device)
        nvectors    = self.embedding(nwords).neg()
        # print(out_vecs.size())
        # print(true_vecs.size())
        # print(nvectors.size())
        oloss       = torch.bmm(out_vecs, true_vecs.transpose(1,2))
        oloss       = oloss.sigmoid().log()
        oloss       = oloss.mean(1)
        nloss       = torch.bmm(nvectors, true_vecs.transpose(1,2))
        nloss       = nloss.squeeze().sigmoid().log()
        nloss       = nloss.view(-1, context_size, self.n_negs)
        nloss       = nloss.sum(2).mean(1)
        # print(oloss.size())
        # print(nloss.size())
        return -(oloss + nloss).mean()

class S2S_lstm(nn.Module):
    def __init__(self, vocab_size = 100, embedding_dim=30, hidden_dim = 256):
        super(S2S_lstm, self).__init__()
        #####################################################################################
        self.vocab_size         = vocab_size
        self.embedding_dim      = embedding_dim
        self.hidden_dim         = hidden_dim
        #####################################################################################
        self.embed              = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=None)
        self.bi_lstm_src        = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=1, bias=True,
            batch_first=True, dropout=0.1, bidirectional=True
        ).to(device)
        self.bi_lstm_trg        = nn.LSTM(
            input_size  = 4*self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, bias=True,
            batch_first = True, dropout=0.1, bidirectional=True
        ).to(device)
        self.projection         = nn.Linear(2*self.hidden_dim, self.embedding_dim)
        #####################################################################################
        self.loss_f             = SGNS(self.embed, vocab_size=vocab_size, n_negs=20).to(device)
        #####################################################################################
    def forward(self, src_tokens, trg_tokens):
        # print(src_tokens.size())
        # print(trg_tokens.size())
        src_embeds                  = self.embed(src_tokens)
        # print(src_embeds.size())
        src_contextual, (h_n, c_n)  = self.bi_lstm_src(src_embeds)
        # print(src_contextual.size())
        # print(h_n.size())
        # print(c_n.size())
        hidden_concat               = torch.cat([h_n[0], h_n[1]], dim=1)
        trg_input                   = torch.cat([hidden_concat.unsqueeze(1).expand_as(src_contextual), src_contextual], dim=-1)
        # print(hidden_concat.size())
        # print(trg_input.size())
        trg_contextual, (h_n, c_n)  = self.bi_lstm_trg(trg_input)
        # print(trg_contextual.size())
        out_vecs                    = self.projection(trg_contextual)
        # print(out_vecs.size())
        loss_                       = self.loss_f(
            src_embeds.reshape(-1, 1, self.embedding_dim),
            out_vecs.reshape(-1, 1, self.embedding_dim)
        )
        print(loss_)
        return loss_

b_size          = 64
vocab_size      = 100
embedding_dim   = 30
hidden_dim      = 100
timesteps       = 50

model           = S2S_lstm(vocab_size = vocab_size, embedding_dim=embedding_dim, hidden_dim = hidden_dim).to(device)

src_tokens  = torch.LongTensor(b_size, timesteps).random_(0, vocab_size).to(device)
trg_tokens  = torch.LongTensor(b_size, timesteps).random_(0, vocab_size).to(device)

optim = Adam(model.parameters())
for i in range(10):
    loss = model(src_tokens, trg_tokens)
    optim.zero_grad()
    loss.backward()
    optim.step()









