
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
    def forward(self, batch_x1, batch_x2, batch_y, batch_features):
        batch_x1        = autograd.Variable(torch.FloatTensor(batch_x1),        requires_grad=False)
        batch_x2        = autograd.Variable(torch.FloatTensor(batch_x2),        requires_grad=False)
        batch_y         = autograd.Variable(torch.FloatTensor(batch_y),         requires_grad=False)
        batch_features  = autograd.Variable(torch.FloatTensor(batch_features),  requires_grad=False)
        if(use_cuda):
            batch_x1        = batch_x1.cuda()
            batch_x2        = batch_x2.cuda()
            batch_y         = batch_y.cuda()
            batch_features  = batch_features.cuda()
        #
        q_context       = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context       = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)

        #
        #
        loss1               = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output, gs_emits, bs_emits


use_cuda = True

embedding_dim       = 30
additional_feats    = 8

model = BCNN(
    embedding_dim       = embedding_dim,
    additional_feats    = additional_feats,
    convolution_size    = 4
)





























