
import json, torch, re, pickle, random, os, sys
from pytorch_transformers import BertModel, BertTokenizer
import  torch.nn as nn
import  torch.optim             as optim
import  torch.nn.functional     as F
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm
import numpy as np

from torch import LongTensor as LT
from torch import FloatTensor as FT

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
    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()

class S2S_lstm(nn.Module):
    def __init__(self, vocab_size = 100, embedding_dim=30, hidden_dim = 256):
        super(S2S_lstm, self).__init__()
        #####################################################################################
        self.vocab_size         = vocab_size
        self.embedding_dim      = embedding_dim
        self.hidden_dim         = hidden_dim
        #####################################################################################
        self.embed              = nn.Embedding(self.vocab_size_from, self.embedding_dim, padding_idx=None)
        self.bi_lstm_src        = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=1, bias=True,
            batch_first=True, dropout=0.1, bidirectional=True
        ).to(device)
        self.bi_lstm_trg        = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=1, bias=True,
            batch_first=True, dropout=0.1, bidirectional=True
        ).to(device)
        #####################################################################################
    def forward(self, doc_vectors):
        l1 = F.leaky_relu(self.layer1(doc_vectors))
        l2 = F.leaky_relu(self.layer2(l1))
        return l2




model       = S2S_lstm(vocab_size_from = 100, vocab_size_to = 100, embedding_dim=30, hidden_dim = 256).to(device)













