#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

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
import  nltk
import  math

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
softmax     = lambda z: np.exp(z) / np.sum(np.exp(z))
stopwords   = nltk.corpus.stopwords.words("english")

use_cuda    = torch.cuda.is_available()
device      = torch.device("cuda") if(use_cuda) else torch.device("cpu")

class HAR_Modeler(nn.Module):
    def __init__(self, embedding_dim = 30,  gru_hidden = 5):
        super(HAR_Modeler, self).__init__()
        #
        self.H          = 2 * gru_hidden
        self.q_h0       = autograd.Variable(torch.randn(2, 1, gru_hidden)).to(device)
        self.d_h0       = autograd.Variable(torch.randn(2, 1, gru_hidden)).to(device)
        self.bigru_q    = nn.GRU(input_size=embedding_dim, hidden_size=gru_hidden, bidirectional=True, batch_first=False).to(device)
        self.bigru_d    = nn.GRU(input_size=embedding_dim, hidden_size=gru_hidden, bidirectional=True, batch_first=False).to(device)
        self.wc         = torch.nn.Parameter(torch.randn(1, 3*self.H))
        #
    def forward(self, doc_sents_embeds, question_embeds):
        q_context           = self.bigru_q(question_embeds.unsqueeze(1), self.q_h0)[0].squeeze(1)
        d_contexts          = [self.bigru_d(sent_embs.unsqueeze(1), self.d_h0)[0].squeeze(1) for sent_embs in doc_sents_embeds]
        for sent_embed in d_contexts:
            cols = []
            for uyq in q_context:
                row = []
                for uxid in sent_embed:
                    concated = torch.cat((uyq, uxid, uyq * uxid))
                    sxy = torch.mm(self.wc, concated.unsqueeze(-1))
                    row.append(sxy)
                row = torch.cat(row).squeeze()
                cols.append(row)
            sxy     = torch.stack(cols).transpose(0,1)
            ########
            sd2q    = F.softmax(sxy, dim=1)
            sq2d    = F.softmax(sxy, dim=0)
            ########
            ad2q    = torch.mm(sd2q, q_context)
            aq2d    = torch.mm(sd2q, sq2d.transpose(0, 1))
            aq2d    = torch.mm(aq2d, sent_embed)
            ########




embedding_dim = 30
doc_sents_embeds   = [
    torch.randn((5, embedding_dim)).to(device),
    torch.randn((4, embedding_dim)).to(device),
    torch.randn((7, embedding_dim)).to(device),
    torch.randn((10, embedding_dim)).to(device),
]
question_embeds     = torch.randn((10, embedding_dim)).to(device)


model = HAR_Modeler(embedding_dim=embedding_dim).to(device)
model(doc_sents_embeds, question_embeds)





