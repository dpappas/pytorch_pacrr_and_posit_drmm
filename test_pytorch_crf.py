

import torch
from torchcrf import CRF

num_tags    = 2  # number of tags is 5
model       = CRF(num_tags)

seq_length = 3  # maximum sequence length in a batch
batch_size = 1  # number of samples in the batch
emissions   = torch.randn(seq_length, batch_size, num_tags)
tags        = torch.tensor([[0], [1], [1]], dtype=torch.long)  # (seq_length, batch_size)
model(emissions, tags)

model.decode(emissions)

'''
or 
https://github.com/mtreviso/linear-chain-crf/blob/master/bilstm_crf.py
'''
