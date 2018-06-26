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
from joblib import Parallel, delayed

cudnn.benchmark = True
torch.manual_seed(my_seed)
print(torch.get_num_threads())
print(torch.cuda.is_available())
print(torch.cuda.device_count())

gpu_device = 0
use_cuda = torch.cuda.is_available()
if(use_cuda):
    torch.cuda.manual_seed(my_seed)

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

class Posit_Drmm_Modeler(nn.Module):
    def __init__( self, nof_filters, filters_size, pretrained_embeds, k_for_maxpool):
        super(Posit_Drmm_Modeler, self).__init__()
        #
        self.nof_sent_filters                       = nof_filters           # number of filters for the convolution of sentences
        self.sent_filters_size                      = filters_size          # The size of the ngram filters we will apply on sentences
        self.nof_quest_filters                      = nof_filters           # number of filters for the convolution of the question
        self.quest_filters_size                     = filters_size          # The size of the ngram filters we will apply on question
        self.k                                      = k_for_maxpool         # k is for the average k pooling
        #
        # if we have a matrix of pretrained embeddings we load it into an Embedding Layer
        self.vocab_size                             = pretrained_embeds.shape[0]
        self.embedding_dim                          = pretrained_embeds.shape[1]
        self.word_embeddings                        = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeds))
        self.word_embeddings.weight.requires_grad   = False
        #
        # We create randomly initialized convolutional filters for the sentences
        self.sent_filters_conv  = torch.nn.Parameter(
            torch.randn(
                self.nof_sent_filters,
                1,                          # the number of is channels is one
                self.sent_filters_size,
                self.embedding_dim
            )
        )
        #
        # We use the same filters for sentences and questions
        self.quest_filters_conv = self.sent_filters_conv
        #
        # a linear function (MLP) that we will apply on the Doc-Aware Query Term Encodings
        self.linear_per_q       = nn.Linear(6, 1, bias=True)
        #
        # Our loss is Binary CrossEntropy loss.
        self.bce_loss           = torch.nn.BCELoss()
        #
        # if we have a gpu we move any function on the gpu.
        if(use_cuda):
            # self.word_embeddings.cuda(gpu_device)
            self.linear_per_q   = self.linear_per_q.cuda(gpu_device)
            self.bce_loss       = self.bce_loss.cuda(gpu_device)
    def one_hot_sim_matrix_one_sent(self, sentence, question):
        '''
        This function takes forever. I am not using it. I just leave it here.
        It is used along with the function get_one_hot_sim_matrix
        :param sentence:    The sentence's indices
        :param question:    The question's indices
        :return:            The one hot similarity matrix between the sentence and the question
        '''
        ret = torch.zeros(sentence.size(0), question.size(0), sentence.size(1))
        for k in range(question.size(0)):
            # if(question[k].data[0]>0):
            if(question[k].item()>0):
                for i in range(sentence.size(0)):
                    for j in range(sentence.size(1)):
                            # if(sentence[i,j] == question[k]).data[0] == True:
                            if(sentence[i,j] == question[k]).item() == True:
                                ret[i][k][j] += 1.
        return ret
    def get_one_hot_sim_matrix(self, sentences, question):
        '''
        This function takes forever. I am not using it. I just leave it here.
        It uses the former one multiple times (one for each sentence in the document).
        :param sentences:
        :param question:
        :return:
        '''
        ret = []
        for i in range(sentences.size(0)):
            ret.append(self.one_hot_sim_matrix_one_sent(sentences[i], question[i]))
        ret = torch.stack(ret)
        ret = ret.transpose(0,1)
        ret = autograd.Variable(ret, requires_grad=False)
        if(use_cuda):
            ret = ret.cuda(gpu_device)
        return ret
    def apply_convolution(self, the_input, the_filters, filter_size):
        '''
        :param the_input:   The matrix that we will apply the filters on.
        :param the_filters: The filters we will apply on the input.
        :param filter_size: Since we operate on a text i pad the sequence using this parameter.
        :return:            The output of the convolution
        '''
        conv_res = F.conv2d(
            the_input.unsqueeze(1),
            the_filters,
            bias        = None,
            stride      = 1,
            padding     = (int(filter_size/2)+1, 0)             # pad the sequence
        )
        conv_res = conv_res[:, :, -1*the_input.size(1):, :]     # just take the output of the text convolution ignoring the first rows that came from the padding
        conv_res = conv_res.squeeze(-1).transpose(1,2)          # transpose to get an output of (batch_size, nof_sents, ... )
        return conv_res
    def pooling_method(self, sim_matrix):
        '''
        Returns two numbers i.e. the maximum value concatenated to the average of the k-maximum values of the input
        :param sim_matrix:      just an input matrix. In our case is a similarity matrix
        :return:
        '''
        sorted_res              = torch.sort(sim_matrix, -1)[0]             # sort the input minimum to maximum
        k_max_pooled            = sorted_res[:,:,:,-self.k:]                # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)        # average these k values
        the_maximum             = k_max_pooled[:, :, :, -1]                 # select the maximum value of each instance
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled], dim=-1) # concatenate maximum value and average of k-max values
        return the_concatenation     # return the concatenation
    def my_cosine_sim(self,A,B):
        '''
        Computes the cosine similarity of A and B
        :param A:
        :param B:
        :return:
        '''
        A_mag       = torch.norm(A, 2, dim=2)
        B_mag       = torch.norm(B, 2, dim=2)
        num         = torch.bmm(A, B.transpose(-1,-2))
        den         = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1,-2))
        dist_mat    = num / den
        return dist_mat
    def apply_masks_on_similarity(self, sentences, question, similarity):
        sim_mask1                = (sentences > 1).float().transpose(0,1).unsqueeze(2).expand_as(similarity)
        sim_mask2                = (question > 1).float().unsqueeze(0).unsqueeze(-1).expand_as(similarity)
        return similarity*sim_mask1*sim_mask2
    def forward(self, sentences, question, target_sents, target_docs):
        #
        # The inputs are numpy arrays so i transform them in the appropriate format
        sentences               = autograd.Variable(torch.LongTensor(sentences), requires_grad=False)
        question                = autograd.Variable(torch.LongTensor(question), requires_grad=False)
        target_sents            = autograd.Variable(torch.LongTensor(target_sents), requires_grad=False)
        target_docs             = autograd.Variable(torch.LongTensor(target_docs), requires_grad=False)
        #
        # Move the data to the gpu if i can use one.
        if(use_cuda):
            sentences           = sentences.cuda(gpu_device)
            question            = question.cuda(gpu_device)
            target_sents        = target_sents.cuda(gpu_device)
            target_docs         = target_docs.cuda(gpu_device)
        #
        # get the question embeddings
        question_embeds         = self.word_embeddings(question)
        #
        # get the sentences embeddings
        sentence_embeds         = self.word_embeddings(sentences.view(sentences.size(0), -1))
        sentence_embeds         = sentence_embeds.view(sentences.size(0), sentences.size(1), sentences.size(2), -1)
        #
        # get the similarity matrix of the "out of context" embeddings. I call it insensitive
        similarity_insensitive  = torch.stack([self.my_cosine_sim(question_embeds, s) for s in sentence_embeds.transpose(0, 1)])
        # when the insensitive score is one, then the embeddings are identical therefore the tokens are the same.
        # So i get the one hot similarity matrix
        similarity_one_hot      = (similarity_insensitive >= (1.0-(1e-05))).float()
        #
        # We apply the convolution on the embeddings to get contextual embeddings
        # Apply it on the question
        q_conv_res              = self.apply_convolution(question_embeds, self.quest_filters_conv, self.quest_filters_size)
        # Apply it on each sentence
        s_conv_res              = torch.stack([self.apply_convolution(s, self.sent_filters_conv, self.sent_filters_size) for s in sentence_embeds])
        # Compute the similarity of the contextual embeddings of the question and each of the sentences.
        similarity_sensitive    = torch.stack([self.my_cosine_sim(q_conv_res, s) for s in s_conv_res.transpose(0, 1)])
        #
        # apply a mask on the similarities because we might have high similarity with UNKN and PAD tokens
        similarity_insensitive  = self.apply_masks_on_similarity(sentences, question, similarity_insensitive)
        similarity_sensitive    = self.apply_masks_on_similarity(sentences, question, similarity_sensitive)
        #
        # print(similarity_insensitive.size())
        # print(similarity_sensitive.size())
        # print(similarity_one_hot.size())
        #
        # Use the pooling function on each one of the similarity matrices. Each one returns 2 values for each sentence
        similarity_insensitive_pooled   = self.pooling_method(similarity_insensitive)
        similarity_sensitive_pooled     = self.pooling_method(similarity_sensitive)
        similarity_one_hot_pooled       = self.pooling_method(similarity_one_hot)
        #
        # Concatenate the pooled features to create a vector of 6 numbers for each sentence.
        similarities_concatenated       = torch.cat([similarity_insensitive_pooled, similarity_sensitive_pooled,similarity_one_hot_pooled],-1)
        similarities_concatenated       = similarities_concatenated.transpose(0,1)
        #
        # Apply an MLP on the concatenated pooled features. Sigmoid(W*X)
        sent_out                        = self.linear_per_q(similarities_concatenated)
        sent_out                        = sent_out.squeeze(-1)
        # sent_out                        = F.relu(sent_out)
        sent_out                        = F.sigmoid(sent_out)
        #
        # Average the output of each sentence just like we did for the document.
        sentence_relevance              = sent_out.sum(-1) / sent_out.size(-1)
        # Compute the loss of the sentences
        sentences_average_loss          = self.bce_loss(sentence_relevance, target_sents.float())
        #
        # We define the document relevance as the maximum value of sentences emitions
        document_emitions               = sentence_relevance.max(-1)[0]
        # Compute the loss of the documents
        document_average_loss           = self.bce_loss(document_emitions, target_docs.float())
        #
        # Average the two computed losses (sentences average loss and document average loss)
        total_loss                      = (sentences_average_loss + document_average_loss) / 2.0
        #
        return(total_loss, sentence_relevance, document_emitions) # return the general loss, the sentences' relevance score and the documents' relevance scores

# We create some random dummy data fo testing
vocab_size          = 40000
# print(vocab_size)
b_size              = 20
max_sents           = 15
max_sent_tokens     = 50
max_quest_tokens    = 30
emb_size            = 100
matrix              = np.random.rand(vocab_size, emb_size)
b_sents             = np.random.randint(1, vocab_size, size=(b_size, max_sents, max_sent_tokens))
b_quest             = np.random.randint(1, vocab_size, size=(b_size, max_quest_tokens))
b_tar_sent          = np.zeros((b_size, max_sents))
b_tar_sent[:10,:5]  = 1.0
b_tar_docs          = np.zeros((b_size))
b_tar_docs[:10]     = 1.0
#

# create the model
k_for_maxpool       = 3
nof_cnn_filters     = 10
filters_size        = 3         # n-gram convolution
model               = Posit_Drmm_Modeler(
    nof_filters         = nof_cnn_filters,
    filters_size        = filters_size,
    pretrained_embeds   = matrix,
    k_for_maxpool       = k_for_maxpool
)
# model.to(device)

if(use_cuda):
    # model = torch.nn.DataParallel(model).cuda()
    model.cuda(gpu_device)

lr          = 0.1
# freeze the embeddings
params      = list(set(model.parameters()) - set([model.word_embeddings.weight]))
# params      = list(set(model.parameters()))
optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# feed the same batch for some "epochs" to see if the model converges
for i in range(2000):
    optimizer.zero_grad()
    cost_, sent_ems, doc_ems = model(
        sentences   = b_sents,
        question    = b_quest,
        target_sents= b_tar_sent,
        target_docs = b_tar_docs
    )
    cost_.backward()
    optimizer.step()
    the_cost = cost_.cpu().item()
    print(the_cost)


'''
1. use mapping for zeros, since PAD similarities will always be one 
2. also similarities and exact matching should be precomputed since we have unknown words
'''
