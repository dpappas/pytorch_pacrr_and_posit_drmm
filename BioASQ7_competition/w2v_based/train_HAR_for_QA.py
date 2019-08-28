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
    def __init__(self, embedding_dim = 30, H = 10):
        super(HAR_Modeler, self).__init__()
        #
        self.q_h0       = autograd.Variable(torch.randn(2, 1, H)).to(device)
        self.d_h0       = autograd.Variable(torch.randn(2, 1, H)).to(device)
        self.bigru_q    = nn.GRU(input_size=embedding_dim, hidden_size=H, bidirectional=True, batch_first=False).to(device)
        self.bigru_d    = nn.GRU(input_size=embedding_dim, hidden_size=H, bidirectional=True, batch_first=False).to(device)
        self.wc         = torch.nn.Parameter(torch.randn(1, 3*H))
        #
    def forward(self, doc_sents_embeds, question_embeds):
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds), requires_grad=False).to(device)
        q_context, hn       = self.bigru_q(question_embeds.unsqueeze(1), self.q_h0)
        d_contexts          = [self.bigru_d(sent_embs.unsqueeze(1), self.d_h0)[0] for sent_embs in doc_sents_embeds]
        print(q_context.size())
        for s in d_contexts:
            print(s.size())


embedding_dim = 30
doc_sents_embeds   = [
    torch.randn((5, embedding_dim)),
    torch.randn((4, embedding_dim)),
    torch.randn((7, embedding_dim)),
    torch.randn((10, embedding_dim)),
]
question_embeds     = torch.randn((10, embedding_dim))


model = HAR_Modeler(embedding_dim=embedding_dim).to(device)
model(doc_sents_embeds, question_embeds)





