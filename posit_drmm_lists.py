#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from gensim.topic_coherence.segmentation import s_one_one

__author__ = 'Dimitris'

my_seed = 1989
# import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_curve, auc
import cPickle as pickle
import numpy as np
import random
import os
random.seed(my_seed)
torch.manual_seed(my_seed)
print(torch.get_num_threads())

def get_the_metrics(gold_labels, predictions):
    if(len(predictions.shape) == 2):
        preds = predictions[:,1]
    else:
        preds = predictions
    false_positive_rate, recall, thresholds = roc_curve(gold_labels, preds)
    roc_auc = auc(false_positive_rate, recall)
    return { 'roc_auc' : roc_auc }

def dummy_test():
    for epoch in range(20):
        dd = pickle.load(open('1.p', 'rb'))
        optimizer.zero_grad()
        cost_, sent_ems, doc_ems = model(
            sentences            = dd['sent_inds'],
            question             = dd['quest_inds'],
            target_sents         = dd['sent_labels'],
            target_docs          = dd['doc_labels'],
            similarity_one_hot   = dd['sim_matrix']
        )
        cost_.backward()
        optimizer.step()
        the_cost = cost_.cpu().item()
        print(the_cost)
    print(20 * '-')

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

def save_checkpoint(epoch, model, min_dev_loss, optimizer, filename='checkpoint.pth.tar'):
    '''
    :param state:       the stete of the pytorch mode
    :param filename:    the name of the file in which we will store the model.
    :return:            Nothing. It just saves the model.
    '''
    state = {
        'epoch':            epoch,
        'state_dict':       model.state_dict(),
        'best_valid_score': min_dev_loss,
        'optimizer':        optimizer.state_dict(),
    }
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
        return [self.word_embeddings(item)for item in items]
    def apply_convolution(self, listed_inputs, the_filters):
        ret             = []
        filter_size     = the_filters.size(2)
        for the_input in listed_inputs:
            the_input   = the_input.unsqueeze(0)
            conv_res    = F.conv2d(the_input.unsqueeze(1), the_filters, bias=None, stride=1, padding=(int(filter_size/2)+1, 0))
            conv_res    = conv_res[:, :, -1*the_input.size(1):, :]
            conv_res    = conv_res.squeeze(-1).transpose(1,2)
            ret.append(conv_res.squeeze(0))
            # ret.append(conv_res)
        return ret
    def my_cosine_sim(self,A,B):
        A           = A.unsqueeze(0)
        B           = B.unsqueeze(0)
        A_mag       = torch.norm(A, 2, dim=2)
        B_mag       = torch.norm(B, 2, dim=2)
        num         = torch.bmm(A, B.transpose(-1,-2))
        den         = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1,-2))
        dist_mat    = num / den
        return dist_mat
    def my_cosine_sim_many(self, quest, sents):
        ret = []
        for sent in sents:
            ret.append(self.my_cosine_sim(quest,sent).squeeze(0))
        return ret
    def pooling_method(self, sim_matrix):
        sorted_res              = torch.sort(sim_matrix, -1)[0]             # sort the input minimum to maximum
        k_max_pooled            = sorted_res[:,-self.k:]                # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)        # average these k values
        the_maximum             = k_max_pooled[:, -1]                 # select the maximum value of each instance
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled], dim=-1) # concatenate maximum value and average of k-max values
        return the_concatenation     # return the concatenation
    def get_sent_output(self, similarity_one_hot_pooled, similarity_insensitive_pooled,similarity_sensitive_pooled):
        ret = []
        for bi in range(len(similarity_one_hot_pooled)):
            ret_r = []
            for j in range(len(similarity_one_hot_pooled[bi])):
                temp = torch.cat(
                    [
                        similarity_insensitive_pooled[bi][j],
                        similarity_sensitive_pooled[bi][j],
                        similarity_one_hot_pooled[bi][j]
                    ],
                    -1
                )
                # print(temp.size())
                lo = self.linear_per_q(temp).squeeze(-1)
                lo = F.sigmoid(lo)
                # lo =  F.hardtanh(lo, min_val=0, max_val=1)
                sr = lo.sum(-1) / lo.size(-1)
                ret_r.append(sr)
            ret.append(torch.stack(ret_r))
        return ret
    def compute_sent_average_loss(self, sent_output, target_sents):
        sentences_average_loss = None
        for i in range(len(sent_output)):
            sal = self.bce_loss(sent_output[i], target_sents[i].float())
            if(sentences_average_loss is None):
                sentences_average_loss  = sal / float(len(sent_output))
            else:
                sentences_average_loss += sal / float(len(sent_output))
        return sentences_average_loss
    def apply_masks_on_similarity(self, sentences, question, similarity):
        for bi in range(len(sentences)):
            qq = question[bi]
            qq = ( qq > 1).float()
            for si in range(len(sentences[bi])):
                ss  = sentences[bi][si]
                ss  = (ss > 1).float()
                sim_mask1 = qq.unsqueeze(-1).expand_as(similarity[bi][si])
                sim_mask2 = ss.unsqueeze(0).expand_as(similarity[bi][si])
                similarity[bi][si] *= sim_mask1
                similarity[bi][si] *= sim_mask2
        return similarity
    def forward(self, sentences, question, target_sents, target_docs, similarity_one_hot):
        #
        question                = [autograd.Variable(torch.LongTensor(item), requires_grad=False) for item in question]
        sentences               = [[autograd.Variable(torch.LongTensor(item), requires_grad=False) for item in item2] for item2 in sentences]
        #
        target_sents            = [autograd.Variable(torch.LongTensor(ts), requires_grad=False) for ts in target_sents]
        target_docs             = autograd.Variable(torch.LongTensor(target_docs), requires_grad=False)
        #
        question_embeds         = self.get_embeds(question)
        q_conv_res              = self.apply_convolution(question_embeds, self.quest_filters_conv)
        #
        sents_embeds            = [self.get_embeds(s) for s in sentences]
        s_conv_res              = [self.apply_convolution(s, self.sent_filters_conv) for s in sents_embeds]
        #
        similarity_insensitive  = [self.my_cosine_sim_many(question_embeds[i], sents_embeds[i]) for i in range(len(sents_embeds))]
        similarity_insensitive  = self.apply_masks_on_similarity(sentences, question, similarity_insensitive)
        similarity_sensitive    = [self.my_cosine_sim_many(q_conv_res[i], s_conv_res[i]) for i in range(len(q_conv_res))]
        similarity_one_hot      = [[autograd.Variable(torch.FloatTensor(item).transpose(0,1), requires_grad=False) for item in item2] for item2 in similarity_one_hot]
        #
        similarity_insensitive_pooled   = [[self.pooling_method(item) for item in item2] for item2 in similarity_insensitive]
        similarity_sensitive_pooled     = [[self.pooling_method(item) for item in item2] for item2 in similarity_sensitive]
        similarity_one_hot_pooled       = [[self.pooling_method(item) for item in item2] for item2 in similarity_one_hot]
        #
        sent_output             = self.get_sent_output(similarity_one_hot_pooled, similarity_insensitive_pooled, similarity_sensitive_pooled)
        sentences_average_loss  = self.compute_sent_average_loss(sent_output, target_sents)
        #
        document_emitions       = torch.stack([ s.max(-1)[0] for s in sent_output])
        document_average_loss   = self.bce_loss(document_emitions, target_docs.float())
        total_loss              = (sentences_average_loss + document_average_loss) / 2.0
        #
        return(total_loss, sent_output, document_emitions) # return the general loss, the sentences' relevance score and the documents' relevance scores

nof_cnn_filters = 12
filters_size    = 3
# matrix          = np.random.random((2900000, 10))
print('LOADING embedding_matrix (14GB)')
# matrix = pickle.load(open('/home/dpappas/joint_task_list_batches/embedding_matrix.p','rb'))
# with h5py.File('/home/dpappas/joint_task_list_batches/embedding_matrix.h5', 'r') as hf:
#     matrix = hf['embeddings'][:]
matrix = np.load('/home/dpappas/joint_task_list_batches/embedding_matrix.npy')
print('Done')

k_for_maxpool   = 5
model           = Posit_Drmm_Modeler(nof_filters=nof_cnn_filters, filters_size=filters_size, pretrained_embeds=matrix, k_for_maxpool=k_for_maxpool)
lr              = 0.01
params          = list(set(model.parameters()) - set([model.word_embeddings.weight]))
optimizer       = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

del(matrix)

# dummy_test()
# exit()

def train_one_epoch(paths, model, optimizer, epoch):
    cost_sum        = 0.0
    average_cost    = 1000.0
    for i in range(len(paths)):
        dd = pickle.load(open(paths[i], 'rb'))
        optimizer.zero_grad()
        cost_, sent_ems, doc_ems = model(
            sentences            = dd['sent_inds'],
            question             = dd['quest_inds'],
            target_sents         = dd['sent_labels'],
            target_docs          = dd['doc_labels'],
            similarity_one_hot   = dd['sim_matrix']
        )
        cost_.backward()
        optimizer.step()
        the_cost        =   cost_.cpu().item()
        cost_sum        +=  the_cost
        average_cost    =   cost_sum / (i+1.0)
        print("\rtrain epoch:{}, batch:{}/{}, aver_loss:{}, batch_loss".format(epoch + 1,i+1,len(paths),average_cost,the_cost), end="")
    print('')
    return average_cost

def test_one_epoch(paths, model, epoch):
    cost_sum        = 0.0
    average_cost    = 1000.0
    for i in range(len(paths)):
        dd = pickle.load(open(paths[i], 'rb'))
        cost_, sent_ems, doc_ems = model(
            sentences            = dd['sent_inds'],
            question             = dd['quest_inds'],
            target_sents         = dd['sent_labels'],
            target_docs          = dd['doc_labels'],
            similarity_one_hot   = dd['sim_matrix']
        )
        the_cost        =   cost_.cpu().item()
        cost_sum        +=  the_cost
        average_cost    =   cost_sum / (i+1.0)
        print("\rtrain epoch:{}, batch:{}/{}, aver_loss:{}, batch_loss".format(epoch + 1,i+1,len(paths),average_cost,the_cost), end="")
    print('')
    return average_cost

dir_with_batches    = '/home/dpappas/joint_task_list_batches/train/'
all_train_paths     = [ dir_with_batches+fpath for fpath  in os.listdir(dir_with_batches) ]
dir_with_batches    = '/home/dpappas/joint_task_list_batches/dev/'
all_dev_paths       = [ dir_with_batches+fpath for fpath  in os.listdir(dir_with_batches) ]
dir_with_batches    = '/home/dpappas/joint_task_list_batches/test/'
all_test_paths      = [ dir_with_batches+fpath for fpath  in os.listdir(dir_with_batches) ]
min_dev_loss        = 10e10
min_loss_epoch      = -1
test_average_loss   = 10e10
odir                = './'
for epoch in range(20):
    train_average_loss      = train_one_epoch(all_train_paths, model, optimizer, epoch)
    dev_average_loss        = test_one_epoch(all_dev_paths, model, epoch)
    if(dev_average_loss < min_dev_loss):
        min_dev_loss        = dev_average_loss
        min_loss_epoch      = epoch+1
        test_average_loss   = test_one_epoch(all_test_paths, model, epoch)
        save_checkpoint(epoch, model, min_dev_loss, optimizer, filename=odir+'best_checkpoint.pth.tar')
    print("epoch:{}, train_average_loss:{}, dev_average_loss:{}, test_average_loss:{}".format(epoch+1, train_average_loss, dev_average_loss, test_average_loss))
    print(20 * '-')

'''
from tqdm import tnrange, tqdm_notebook, tqdm
dir_with_batches = '/home/dpappas/joint_task_list_batches/'
all_paths = [ dir_with_batches+fpath for fpath  in os.listdir(dir_with_batches) ]
for epoch in tqdm(range(20), desc='epochs'):
    for i in tqdm(range(len(all_paths))):
        dd = pickle.load(open(all_paths[i], 'rb'))
        optimizer.zero_grad()
        cost_, sent_ems, doc_ems = model(
            sentences            = dd['sent_inds'],
            question             = dd['quest_inds'],
            target_sents         = dd['sent_labels'],
            target_docs          = dd['doc_labels'],
            similarity_one_hot   = dd['sim_matrix']
        )
        cost_.backward()
        optimizer.step()
        the_cost = cost_.cpu().item()
        print(the_cost)
    print(20 * '-')
'''


