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

# y = torch.rand(1, 10)
y = autograd.Variable(torch.FloatTensor([0.2, 0.7, 0.8, 0.1]))
y1 = torch.sigmoid(y)
y2 = F.softmax(y, dim=-1)
y3 = F.softmax(y1, dim=-1)

print(y)
print(y/torch.sum(y))
print(y1/torch.sum(y1))
print(y1/y1.size(-1))
print(y2)
print(y3)
# print(F.softmax(y/torch.sum(y), dim=-1))


