#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = 'Dimitris'

my_seed = 1989
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import cPickle as pickle
import numpy as np
import random
random.seed(my_seed)
torch.manual_seed(my_seed)
print(torch.get_num_threads())

def loadGloveModel(w2v_voc, w2v_vec):
    '''
    :param w2v_voc: the txt file with the vocabulary extracted from gensim
    :param w2v_vec: the txt file with the vectors extracted from gensim
    :return: vocab is a python dictionary with the indices of each word. matrix is a numpy matrix with all the vectors.
             PAD is special token for padding to maximum length. It has a vector of zeros.
             UNK is special token for any token not found in the vocab. It has a vector equal to the average of all other vectors.
    '''
    temp_vocab  = pickle.load(open(w2v_voc,'rb'))
    temp_matrix = pickle.load(open(w2v_vec,'rb'))
    print("Loading Glove Model")
    vocab, matrix   = {}, []
    vocab['PAD']    = 0
    vocab['UNKN']   = len(vocab)
    for i in range(len(temp_vocab)):
        matrix.append(temp_matrix[i])
        vocab[temp_vocab[i]] = len(vocab)
    matrix          = np.array(matrix)
    av              = np.average(matrix,0)
    pad             = np.zeros(av.shape)
    matrix          = np.vstack([pad, av, matrix])
    print("Done.",len(vocab)," words loaded!")
    return vocab, matrix

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    '''
    :param state:       the stete of the pytorch mode
    :param filename:    the name of the file in which we will store the model.
    :return:            Nothing. It just saves the model.
    '''
    torch.save(state, filename)

def print_params(model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(model)
    print(40 * '=')
    total_params = 0
    for parameter in model.parameters():
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        total_params += v
    print(40 * '=')
    print(total_params)
    print(40 * '=')

dd = pickle.load(open('1.p','rb'))

# dd['doc_labels'][0]
# dd['sent_labels'][0]
# dd['quest_inds'][0]
# len(dd['sent_inds'][0])
# dd['sim_matrix'][0][0].shape

class Posit_Drmm_Modeler(nn.Module):
    def __init__(self, nof_filters, filters_size, pretrained_embeds, k_for_maxpool):
        super(Posit_Drmm_Modeler, self).__init__()
        self.nof_sent_filters                       = nof_filters           # number of filters for the convolution of sentences
        self.sent_filters_size                      = filters_size          # The size of the ngram filters we will apply on sentences
        self.nof_quest_filters                      = nof_filters           # number of filters for the convolution of the question
        self.quest_filters_size                     = filters_size          # The size of the ngram filters we will apply on question
        self.k                                      = k_for_maxpool         # k is for the average k pooling
        self.vocab_size                             = pretrained_embeds.shape[0]
        self.embedding_dim                          = pretrained_embeds.shape[1]
        self.word_embeddings                        = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeds))
        self.word_embeddings.weight.requires_grad   = False
        self.sent_filters_conv  = torch.nn.Parameter(torch.randn(self.nof_sent_filters,1,self.sent_filters_size,self.embedding_dim))
        self.quest_filters_conv = self.sent_filters_conv
        self.linear_per_q       = nn.Linear(6, 1, bias=True)
        self.bce_loss           = torch.nn.BCELoss()
    def get_embeds(self, items):
        return [
            self.word_embeddings(
                autograd.Variable(
                    torch.LongTensor(item),
                    requires_grad = False
                )
            )
            for item in items
        ]
    def forward(self,sentences,question,target_sents,target_docs):
        target_sents    = [autograd.Variable(torch.LongTensor(ts), requires_grad=False) for ts in target_sents]
        target_docs     = autograd.Variable(torch.LongTensor(target_docs), requires_grad=False)
        question_embeds = self.get_embeds(question)
        sents_embeds    = [self.get_embeds(s) for s in sentences]
        # print(len(sents_embeds))
        # for t in sents_embeds:
        #     print(len(t))
        exit()
        #
        sentences               = autograd.Variable(torch.LongTensor(sentences), requires_grad=False)
        question                = autograd.Variable(torch.LongTensor(question), requires_grad=False)
        question_embeds         = self.word_embeddings(question)
        sentence_embeds         = self.word_embeddings(sentences.view(sentences.size(0), -1))
        sentence_embeds         = sentence_embeds.view(sentences.size(0), sentences.size(1), sentences.size(2), -1)
        similarity_insensitive  = torch.stack([self.my_cosine_sim(question_embeds, s) for s in sentence_embeds.transpose(0, 1)])
        similarity_one_hot      = (similarity_insensitive >= (1.0-(1e-05))).float()
        q_conv_res              = self.apply_convolution(question_embeds, self.quest_filters_conv, self.quest_filters_size)
        s_conv_res              = torch.stack([self.apply_convolution(s, self.sent_filters_conv, self.sent_filters_size) for s in sentence_embeds])
        similarity_sensitive    = torch.stack([self.my_cosine_sim(q_conv_res, s) for s in s_conv_res.transpose(0, 1)])
        similarity_insensitive  = self.apply_masks_on_similarity(sentences, question, similarity_insensitive)
        similarity_sensitive    = self.apply_masks_on_similarity(sentences, question, similarity_sensitive)
        similarity_insensitive_pooled   = self.pooling_method(similarity_insensitive)
        similarity_sensitive_pooled     = self.pooling_method(similarity_sensitive)
        similarity_one_hot_pooled       = self.pooling_method(similarity_one_hot)
        similarities_concatenated       = torch.cat([similarity_insensitive_pooled, similarity_sensitive_pooled,similarity_one_hot_pooled],-1)
        similarities_concatenated       = similarities_concatenated.transpose(0,1)
        sent_out                        = self.linear_per_q(similarities_concatenated)
        sent_out                        = sent_out.squeeze(-1)
        sent_out                        = F.sigmoid(sent_out)
        sentence_relevance              = sent_out.sum(-1) / sent_out.size(-1)
        sentences_average_loss          = self.bce_loss(sentence_relevance, target_sents.float())
        document_emitions               = sentence_relevance.max(-1)[0]
        document_average_loss           = self.bce_loss(document_emitions, target_docs.float())
        total_loss                      = (sentences_average_loss + document_average_loss) / 2.0
        return(total_loss, sentence_relevance, document_emitions) # return the general loss, the sentences' relevance score and the documents' relevance scores

nof_cnn_filters = 10
filters_size    = 3
matrix          = np.random.random((2900000, 10))
k_for_maxpool   = 5
model               = Posit_Drmm_Modeler(
    nof_filters         = nof_cnn_filters,
    filters_size        = filters_size,
    pretrained_embeds   = matrix,
    k_for_maxpool       = k_for_maxpool
)

lr = 0.1
params      = list(set(model.parameters()) - set([model.word_embeddings.weight]))
optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for i in range(2):
    optimizer.zero_grad()
    cost_, sent_ems, doc_ems = model(
        sentences   = dd['sent_inds'],
        question    = dd['quest_inds'],
        target_sents= dd['sent_labels'],
        target_docs = dd['doc_labels']
    )
    cost_.backward()
    optimizer.step()
    the_cost = cost_.cpu().item()
    print(the_cost)




