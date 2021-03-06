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
from colour import Color
from flask import url_for
from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re, os
import random
import numpy as np
import pickle


bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
softmax     = lambda z: np.exp(z) / np.sum(np.exp(z))
stopwords   = nltk.corpus.stopwords.words("english")

def get_words(s, idf, max_idf):
    sl  = tokenize(s)
    sl  = [s for s in sl]
    sl2 = [s for s in sl if idf_val(s, idf, max_idf) >= 2.0]
    return sl, sl2

def uwords(words):
  uw = {}
  for w in words:
    uw[w] = 1
  return [w for w in uw]

def ubigrams(words):
  uw = {}
  prevw = "<pw>"
  for w in words:
    uw[prevw + '_' + w] = 1
    prevw = w
  return [w for w in uw]

def query_doc_overlap(qwords, dwords, idf, max_idf):
    # % Query words in doc.
    qwords_in_doc = 0
    idf_qwords_in_doc = 0.0
    idf_qwords = 0.0
    for qword in uwords(qwords):
      idf_qwords += idf_val(qword, idf, max_idf)
      for dword in uwords(dwords):
        if qword == dword:
          idf_qwords_in_doc += idf_val(qword, idf, max_idf)
          qwords_in_doc     += 1
          break
    if len(qwords) <= 0:
      qwords_in_doc_val = 0.0
    else:
      qwords_in_doc_val = (float(qwords_in_doc) /
                           float(len(uwords(qwords))))
    if idf_qwords <= 0.0:
      idf_qwords_in_doc_val = 0.0
    else:
      idf_qwords_in_doc_val = float(idf_qwords_in_doc) / float(idf_qwords)
    # % Query bigrams  in doc.
    qwords_bigrams_in_doc = 0
    idf_qwords_bigrams_in_doc = 0.0
    idf_bigrams = 0.0
    for qword in ubigrams(qwords):
      wrds = qword.split('_')
      idf_bigrams += idf_val(wrds[0], idf, max_idf) * idf_val(wrds[1], idf, max_idf)
      for dword in ubigrams(dwords):
        if qword == dword:
          qwords_bigrams_in_doc += 1
          idf_qwords_bigrams_in_doc += (idf_val(wrds[0], idf, max_idf) * idf_val(wrds[1], idf, max_idf))
          break
    if len(qwords) <= 0:
      qwords_bigrams_in_doc_val = 0.0
    else:
      qwords_bigrams_in_doc_val = (float(qwords_bigrams_in_doc) / float(len(ubigrams(qwords))))
    if idf_bigrams <= 0.0:
      idf_qwords_bigrams_in_doc_val = 0.0
    else:
      idf_qwords_bigrams_in_doc_val = (float(idf_qwords_bigrams_in_doc) / float(idf_bigrams))
    return [
        qwords_in_doc_val,
        qwords_bigrams_in_doc_val,
        idf_qwords_in_doc_val,
        idf_qwords_bigrams_in_doc_val
    ]

def GetScores(qtext, dtext, bm25, idf, max_idf):
    qwords, qw2 = get_words(qtext, idf, max_idf)
    dwords, dw2 = get_words(dtext, idf, max_idf)
    qd1         = query_doc_overlap(qwords, dwords, idf, max_idf)
    bm25        = [bm25]
    return qd1[0:3] + bm25

# Compute the term frequency of a word for a specific document
def tf(term, document):
    tf = 0
    for word in document:
        if word == term:
            tf += 1
    if len(document) == 0:
        return tf
    else:
        return tf/len(document)

# Use BM25 ranking function in order to cimpute the similarity score between a question anda snippet
# query: the given question
# document: the snippet
# k1, b: parameters
# idf_scores: list with the idf scores
# avddl: average document length
# nomalize: in case we want to use Z-score normalization (Boolean)
# mean, deviation: variables used for Z-score normalization
def similarity_score(query, document, k1, b, idf_scores, avgdl, normalize, mean, deviation, rare_word):
    score = 0
    for query_term in query:
        if query_term not in idf_scores:
            score += rare_word * (
                    (tf(query_term, document) * (k1 + 1)) /
                    (
                            tf(query_term, document) +
                            k1 * (1 - b + b * (len(document) / avgdl))
                    )
            )
        else:
            score += idf_scores[query_term] * ((tf(query_term, document) * (k1 + 1)) / (tf(query_term, document) + k1 * (1 - b + b * (len(document) / avgdl))))
    if normalize:
        return ((score - mean)/deviation)
    else:
        return score

def snip_is_relevant(one_sent, gold_snips):
    # print one_sent
    # pprint(gold_snips)
    return int(
        any(
            [
                (one_sent.encode('ascii', 'ignore')  in gold_snip.encode('ascii','ignore'))
                or
                (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii','ignore'))
                for gold_snip in gold_snips
            ]
        )
    )

def prep_data(quest, sents, the_bm25, wv, good_snips, idf, max_idf):
    ####
    quest_toks      = tokenize(quest)
    good_doc_af     = GetScores(quest, ' '.join(sents), the_bm25, idf, max_idf)
    good_doc_af.append(len(sents) / 60.)
    #
    doc_toks            = tokenize(' '.join(sents))
    tomi                = (set(doc_toks) & set(quest_toks))
    tomi_no_stop        = tomi - set(stopwords)
    BM25score           = similarity_score(quest_toks, doc_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
    tomi_no_stop_idfs   = [idf_val(w, idf, max_idf) for w in tomi_no_stop]
    tomi_idfs           = [idf_val(w, idf, max_idf) for w in tomi]
    quest_idfs          = [idf_val(w, idf, max_idf) for w in quest_toks]
    features            = [
        len(quest)                                      / 300.,
        len(' '.join(sents))                            / 300.,
        len(tomi_no_stop)                               / 100.,
        BM25score,
        sum(tomi_no_stop_idfs)                          / 100.,
        sum(tomi_idfs)                                  / sum(quest_idfs),
    ]
    good_doc_af.extend(features)
    ####
    good_sents_embeds, good_sents_escores, held_out_sents, good_sent_tags = [], [], [], []
    for good_text in sents:
        sent_toks                   = tokenize(good_text)
        good_tokens, good_embeds    = get_embeds(sent_toks, wv)
        good_escores                = GetScores(quest, good_text, the_bm25, idf, max_idf)[:-1]
        good_escores.append(len(sent_toks)/ 342.)
        if (len(good_embeds) > 0):
            #
            tomi                = (set(sent_toks) & set(quest_toks))
            tomi_no_stop        = tomi - set(stopwords)
            BM25score           = similarity_score(quest_toks, sent_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
            tomi_no_stop_idfs   = [idf_val(w, idf, max_idf) for w in tomi_no_stop]
            tomi_idfs           = [idf_val(w, idf, max_idf) for w in tomi]
            quest_idfs          = [idf_val(w, idf, max_idf) for w in quest_toks]
            features            = [
                len(quest)              / 300.,
                len(good_text)          / 300.,
                len(tomi_no_stop)       / 100.,
                BM25score,
                sum(tomi_no_stop_idfs)  / 100.,
                sum(tomi_idfs)          / sum(quest_idfs),
            ]
            #
            good_sents_embeds.append(good_embeds)
            good_sents_escores.append(good_escores+features)
            held_out_sents.append(good_text)
            good_sent_tags.append(snip_is_relevant(' '.join(bioclean(good_text)), good_snips))
    ####
    return {
        'sents_embeds'     : good_sents_embeds,
        'sents_escores'    : good_sents_escores,
        'doc_af'           : good_doc_af,
        'sent_tags'        : good_sent_tags,
        'held_out_sents'   : held_out_sents,
    }

def print_params(model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(model)
    print(40 * '=')
    trainable       = 0
    untrainable     = 0
    for parameter in model.parameters():
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        if(parameter.requires_grad):
            trainable   += v
        else:
            untrainable += v
    total_params = trainable + untrainable
    print(40 * '=')
    print('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    print(40 * '=')

def load_model_from_checkpoint(resume_from):
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

def tokenize(x):
  return bioclean(x)

def GetWords(data, doc_text, words):
  for i in range(len(data['queries'])):
    qwds = tokenize(data['queries'][i]['query_text'])
    for w in qwds:
      words[w] = 1
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      dtext = (
              doc_text[doc_id]['title'] + ' <title> ' + doc_text[doc_id]['abstractText']
              # +
              # ' '.join(
              #     [
              #         ' '.join(mm) for mm in
              #         get_the_mesh(doc_text[doc_id])
              #     ]
              # )
      )
      dwds = tokenize(dtext)
      for w in dwds:
        words[w] = 1

def load_idfs(idf_path, words):
    print('Loading IDF tables')
    #
    # with open(dataloc + 'idf.pkl', 'rb') as f:
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    ret = {}
    for w in words:
        if w in idf:
            ret[w] = idf[w]
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    idf = None
    print('Loaded idf tables with max idf {}'.format(max_idf))
    #
    return ret, max_idf

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

def get_embeds(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if(tok in wv):
            ret1.append(tok)
            ret2.append(wv[tok])
    return ret1, np.array(ret2, 'float64')

def create_table(tokens1, tokens2, scores):
    ret_html = '<table>'
    ret_html += '<tr><td></td>'
    ####################
    for tok1 in tokens1:
        ret_html += '<th>{}</th>'.format(tok1)
    ret_html += '</tr>'
    ####################
    for i in range(len(tokens2)):
        tok2        = tokens2[i]
        ret_html    += '<tr>'
        ret_html    += '<th>{}</th>'.format(tok2)
        for j in range(len(tokens1)):
            tok1    = tokens1[j]
            score   = int(scores[j][i])
            ret_html += '<td style="min-width:50px" title="{}" score="{}" bgcolor="{}"></div></td>'.format('{} : {} : {}'.format(tok1,tok2,str(score)), score, yellow_colors[score])
        ret_html += '</tr>'
    ret_html += '</table>'
    ####################
    max_scores      = scores.max(axis=1)
    aver_scores     = np.average(scores, axis=1)
    aver5_scores    = np.average(np.sort(scores, axis=1)[:, -5:], axis=1)
    aver5min_scores = np.average(np.sort(scores, axis=1)[:, :5], axis=1)
    # print(max_scores.shape)
    ret_html        += "<p><b>Pooled scores:</b></p>"
    ret_html        += '<table>'
    ret_html        += '<tr><td></td>'
    for tok1 in tokens1:
        ret_html    += '<th>{}</th>'.format(tok1)
    ret_html        += '</tr>'
    ####################
    ret_html        += '<tr><td>Max Score:</td>'
    for j in range(len(tokens1)):
        score       =  int(max_scores[j])
        ret_html    += '<td style="min-width:50px" score="{}" bgcolor="{}"></div></td>'.format(score, yellow_colors[score])
    ####################
    ret_html        += '<tr><td>Aver of max 5 Scores:</td>'
    for j in range(len(tokens1)):
        score       =  int(aver5_scores[j])
        ret_html    += '<td style="min-width:50px" score="{}" bgcolor="{}"></div></td>'.format(score, yellow_colors[score])
    ####################
    ret_html        += '<tr><td>Aver of min 5 Scores:</td>'
    for j in range(len(tokens1)):
        score       =  int(aver5min_scores[j])
        ret_html    += '<td style="min-width:50px" score="{}" bgcolor="{}"></div></td>'.format(score, yellow_colors[score])
    ####################
    ret_html        += '<tr><td>Aver of all Scores:</td>'
    for j in range(len(tokens1)):
        score       =  int(aver_scores[j])
        ret_html    += '<td style="min-width:50px" score="{}" bgcolor="{}"></div></td>'.format(score, yellow_colors[score])
    ####################
    ret_html        += '<tr><td>normalized (divided by max) IDF Score:</td>'
    for j in range(len(tokens1)):
        score       = idf_val(tokens1[j], idf, max_idf)
        score       = int((score/max_idf) * 100)
        ret_html    += '<td style="min-width:50px" score="{}" bgcolor="{}"></div></td>'.format(score, blue_colors[score])
    ####################
    ret_html     += '</tr>'
    ret_html     += '</table>'
    return ret_html

def get_sim(quest_text, sent_text):
    # quest_text  = 'What is the mechanism of action of anlotinib'
    # sent_text   = 'Anlotinib is a novel TKI targeting the vascular endothelial growth factor receptor (VEGFR), fibroblast growth factor receptor (FGFR), platelet-derived growth factor receptor (PDGFR) and c-Kit.'
    sents       = [sent_text]
    quest_tokens, quest_embeds = get_embeds(tokenize(quest_text), wv)
    q_idfs      = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
    datum       = prep_data(quest_text, sents, 2.43, wv, [], idf, max_idf)
    sim_sens    = model.emit_one(
        doc1_sents_embeds   = datum['sents_embeds'],
        question_embeds     = quest_embeds,
        q_idfs              = q_idfs,
        sents_gaf           = datum['sents_escores']
    )
    return sim_sens[0].data.numpy()

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self,
             embedding_dim          = 30,
             k_for_maxpool          = 5,
             sentence_out_method    = 'MLP',
             k_sent_maxpool         = 1
         ):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k                                      = k_for_maxpool
        self.k_sent_maxpool                         = k_sent_maxpool
        self.doc_add_feats                          = 11
        self.sent_add_feats                         = 10
        #
        self.embedding_dim                          = embedding_dim
        self.sentence_out_method                    = sentence_out_method
        # to create q weights
        self.init_context_module()
        self.init_question_weight_module()
        self.init_mlps_for_pooled_attention()
        self.init_sent_output_layer()
        self.init_doc_out_layer()
        # doc loss func
        self.margin_loss        = nn.MarginRankingLoss(margin=1.0)
        if(use_cuda):
            self.margin_loss    = self.margin_loss.cuda()
    def init_mesh_module(self):
        self.mesh_h0    = autograd.Variable(torch.randn(1, 1, self.embedding_dim))
        self.mesh_gru   = nn.GRU(self.embedding_dim, self.embedding_dim)
        if(use_cuda):
            self.mesh_h0    = self.mesh_h0.cuda()
            self.mesh_gru   = self.mesh_gru.cuda()
    def init_context_module(self):
        self.trigram_conv_1             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        # self.trigram_conv_activation_1  = torch.nn.LeakyReLU(negative_slope=0.1)
        self.trigram_conv_activation_1 = torch.nn.Sigmoid()
        self.trigram_conv_2             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        # self.trigram_conv_activation_2  = torch.nn.LeakyReLU(negative_slope=0.1)
        self.trigram_conv_activation_2 = torch.nn.Sigmoid()
        if(use_cuda):
            self.trigram_conv_1             = self.trigram_conv_1.cuda()
            self.trigram_conv_2             = self.trigram_conv_2.cuda()
            self.trigram_conv_activation_1  = self.trigram_conv_activation_1.cuda()
            self.trigram_conv_activation_2  = self.trigram_conv_activation_2.cuda()
    def init_question_weight_module(self):
        self.q_weights_mlp      = nn.Linear(self.embedding_dim+1, 1, bias=True)
        if(use_cuda):
            self.q_weights_mlp  = self.q_weights_mlp.cuda()
    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1      = nn.Linear(3 * 3, 8, bias=True)
        self.my_relu1           = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2      = nn.Linear(8, 1, bias=True)
        if(use_cuda):
            self.linear_per_q1  = self.linear_per_q1.cuda()
            self.linear_per_q2  = self.linear_per_q2.cuda()
            self.my_relu1       = self.my_relu1.cuda()
    def init_sent_output_layer(self):
        if(self.sentence_out_method == 'MLP'):
            self.sent_out_layer_1       = nn.Linear(self.sent_add_feats+1, 8, bias=False)
            self.sent_out_activ_1       = torch.nn.LeakyReLU(negative_slope=0.1)
            self.sent_out_layer_2       = nn.Linear(8, 1, bias=False)
            if(use_cuda):
                self.sent_out_layer_1   = self.sent_out_layer_1.cuda()
                self.sent_out_activ_1   = self.sent_out_activ_1.cuda()
                self.sent_out_layer_2   = self.sent_out_layer_2.cuda()
        else:
            self.sent_res_h0    = autograd.Variable(torch.randn(2, 1, 5))
            self.sent_res_bigru = nn.GRU(input_size=self.sent_add_feats+1, hidden_size=5, bidirectional=True, batch_first=False)
            self.sent_res_mlp   = nn.Linear(10, 1, bias=False)
            if(use_cuda):
                self.sent_res_h0    = self.sent_res_h0.cuda()
                self.sent_res_bigru = self.sent_res_bigru.cuda()
                self.sent_res_mlp   = self.sent_res_mlp.cuda()
    def init_doc_out_layer(self):
        self.final_layer_1 = nn.Linear(self.doc_add_feats+self.k_sent_maxpool, 8, bias=True)
        self.final_activ_1  = torch.nn.LeakyReLU(negative_slope=0.1)
        self.final_layer_2  = nn.Linear(8, 1, bias=True)
        self.oo_layer       = nn.Linear(2, 1, bias=True)
        if(use_cuda):
            self.final_layer_1  = self.final_layer_1.cuda()
            self.final_activ_1  = self.final_activ_1.cuda()
            self.final_layer_2  = self.final_layer_2.cuda()
            self.oo_layer       = self.oo_layer.cuda()
    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta      = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos
    def apply_context_gru(self, the_input, h0):
        output, hn      = self.context_gru(the_input.unsqueeze(1), h0)
        output          = self.context_gru_activation(output)
        out_forward     = output[:, 0, :self.embedding_dim]
        out_backward    = output[:, 0, self.embedding_dim:]
        output          = out_forward + out_backward
        res             = output + the_input
        return res, hn
    def apply_context_convolution(self, the_input, the_filters, activation):
        conv_res        = the_filters(the_input.transpose(0,1).unsqueeze(0))
        if(activation is not None):
            conv_res    = activation(conv_res)
        pad             = the_filters.padding[0]
        ind_from        = int(np.floor(pad/2.0))
        ind_to          = ind_from + the_input.size(0)
        conv_res        = conv_res[:, :, ind_from:ind_to]
        conv_res        = conv_res.transpose(1, 2)
        # residual
        conv_res = conv_res + the_input
        return conv_res.squeeze(0)
    def my_cosine_sim(self, A, B):
        A           = A.unsqueeze(0)
        B           = B.unsqueeze(0)
        A_mag       = torch.norm(A, 2, dim=2)
        B_mag       = torch.norm(B, 2, dim=2)
        num         = torch.bmm(A, B.transpose(-1,-2))
        den         = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1,-2))
        dist_mat    = num / den
        return dist_mat
    def pooling_method(self, sim_matrix):
        sorted_res              = torch.sort(sim_matrix, -1)[0]                             # sort the input minimum to maximum
        k_max_pooled            = sorted_res[:,-self.k:]                                    # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)                        # average these k values
        the_maximum             = k_max_pooled[:, -1]                                       # select the maximum value of each instance
        the_average_over_all    = sorted_res.sum(-1)/float(sim_matrix.size(1))              # add average of all elements as long sentences might have more matches
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled, the_average_over_all], dim=-1)  # concatenate maximum value and average of k-max values
        return the_concatenation     # return the concatenation
    def get_output(self, input_list, weights):
        temp    = torch.cat(input_list, -1)
        lo      = self.linear_per_q1(temp)
        lo      = self.my_relu1(lo)
        lo      = self.linear_per_q2(lo)
        lo      = lo.squeeze(-1)
        lo      = lo * weights
        sr      = lo.sum(-1) / lo.size(-1)
        return sr
    def apply_sent_res_bigru(self, the_input):
        output, hn      = self.sent_res_bigru(the_input.unsqueeze(1), self.sent_res_h0)
        output          = self.sent_res_mlp(output)
        return output.squeeze(-1).squeeze(-1)
    def do_for_one_doc_cnn(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights, k2):
        res     = []
        ret3    = []
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            if(use_cuda):
                sent_embeds     = sent_embeds.cuda()
                gaf             = gaf.cuda()
            conv_res            = self.apply_context_convolution(sent_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
            conv_res            = self.apply_context_convolution(conv_res,      self.trigram_conv_2, self.trigram_conv_activation_2)
            #
            sim_insens          = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_oh              = (sim_insens > (1 - (1e-3))).float()
            sim_sens            = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            ret3.append(sim_sens)
            #
            insensitive_pooled  = self.pooling_method(sim_insens)
            sensitive_pooled    = self.pooling_method(sim_sens)
            oh_pooled           = self.pooling_method(sim_oh)
            #
            sent_emit           = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats      = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        if(self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer_1(res)
            res = self.sent_out_activ_1(res)
            res = self.sent_out_layer_2(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        # ret = self.get_max(res).unsqueeze(0)
        ret = self.get_kmax(res, k2)
        return ret, res, ret3
    def do_for_one_doc_bigru(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights, k2):
        res = []
        hn  = self.context_h0
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            if(use_cuda):
                sent_embeds     = sent_embeds.cuda()
                gaf             = gaf.cuda()
            conv_res, hn        = self.apply_context_gru(sent_embeds, hn)
            #
            sim_insens          = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_oh              = (sim_insens > (1 - (1e-3))).float()
            sim_sens            = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled  = self.pooling_method(sim_insens)
            sensitive_pooled    = self.pooling_method(sim_sens)
            oh_pooled           = self.pooling_method(sim_oh)
            #
            sent_emit           = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats      = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        if(self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer_1(res)
            res = self.sent_out_activ_1(res)
            res = self.sent_out_layer_2(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        # ret = self.get_max(res).unsqueeze(0)
        ret = self.get_kmax(res, k2)
        res = torch.sigmoid(res)
        return ret, res
    def get_max(self, res):
        return torch.max(res)
    def get_kmax(self, res, k):
        res     = torch.sort(res,0)[0]
        res     = res[-k:].squeeze(-1)
        if(len(res.size())==0):
            res = res.unsqueeze(0)
        if(res.size()[0] < k):
            to_concat       = torch.zeros(k - res.size()[0])
            if(use_cuda):
                to_concat   = to_concat.cuda()
            res             = torch.cat([res, to_concat], -1)
        return res
    def get_max_and_average_of_k_max(self, res, k):
        k_max_pooled            = self.get_kmax(res, k)
        average_k_max_pooled    = k_max_pooled.sum()/float(k)
        the_maximum             = k_max_pooled[-1]
        the_concatenation       = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation
    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res
    def get_maxmin_max(self, res):
        res = self.min_max_norm(res)
        res = torch.max(res)
        return res
    def apply_mesh_gru(self, mesh_embeds):
        mesh_embeds             = autograd.Variable(torch.FloatTensor(mesh_embeds), requires_grad=False)
        if(use_cuda):
            mesh_embeds         = mesh_embeds.cuda()
        output, hn              = self.mesh_gru(mesh_embeds.unsqueeze(1), self.mesh_h0)
        return output[-1,0,:]
    def get_mesh_rep(self, meshes_embeds, q_context):
        meshes_embeds   = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds   = torch.stack(meshes_embeds)
        sim_matrix      = self.my_cosine_sim(meshes_embeds, q_context).squeeze(0)
        max_sim         = torch.sort(sim_matrix, -1)[0][:, -1]
        output          = torch.mm(max_sim.unsqueeze(0), meshes_embeds)[0]
        return output
    def emit_one(self, doc1_sents_embeds, question_embeds, q_idfs, sents_gaf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        if(use_cuda):
            q_idfs          = q_idfs.cuda()
            question_embeds = question_embeds.cuda()
        #
        q_context           = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context           = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits, sim_Sens  = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights, self.k_sent_maxpool)
        #
        return sim_Sens

###########################################################
use_cuda = False #torch.cuda.is_available()
w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
###########################################################
avgdl               = 21.1907
mean                = 0.6275
deviation           = 1.2210
print(avgdl, mean, deviation)
###########################################################
k_for_maxpool       = 5
k_sent_maxpool      = 5
embedding_dim       = 30 #200
###########################################################
my_seed     = 1
random.seed(my_seed)
torch.manual_seed(my_seed)
###########################################################
print('Compiling model...')
model       = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool)
if(use_cuda):
    model   = model.cuda()
###########################################################
resume_from     = '/home/dpappas/bioasq_jpdrmm_2L_0p01_run_0/best_dev_checkpoint.pth.tar'
load_model_from_checkpoint(resume_from)
params      = model.parameters()
print_params(model)
###########################################################
print('loading pickle data')
with open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3', 'r') as f:
    bioasq7_data = json.load(f)
    bioasq7_data = dict((q['id'], q) for q in bioasq7_data['questions'])
with open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl', 'rb') as f:
    test_docs = pickle.load(f)
words = {}
GetWords(test_data, test_docs, words)
###########################################################
print('loading idfs')
idf, max_idf = load_idfs(idf_pickle_path, words)
print('loading w2v')
wv = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
wv = dict([(word, wv[word]) for word in wv.vocab.keys() if (word in words)])
###########################################################

white           = Color("white")
yellow_colors   = list(white.range_to(Color("red"), 101))
yellow_colors   = [c.get_hex_l() for c in yellow_colors]
blue_colors     = list(white.range_to(Color("blue"), 101))
blue_colors     = [c.get_hex_l() for c in blue_colors]

app = Flask(__name__)

@app.route("/")
def get_news():
    return render_template("sentence_similarity.html")

@app.route("/test_similarity_matrix", methods=["POST", "GET"])
def test_similarity_matrix():
    sent1           = request.form.get("sent1").strip()
    sent2           = request.form.get("sent2").strip()
    tokens1         = tokenize(sent1)
    tokens2         = tokenize(sent2)
    scores          = get_sim(' '.join(tokens1), ' '.join(tokens2)).clip(min=0) * 100
    #############
    ret_html    = '''
    <html>
    <head>
    <style>
    table, th, td {border: 1px solid black;}
    .floatLeft { width: 50%; float: left; }
    .floatRight {width: 50%; float: right; }
    .container { overflow: hidden; }
    </style>
    </head>
    <body>
    '''
    ret_html    += '<div class="container">'
    ret_html    += '<div class="floatLeft">'
    ret_html    += '<p><h2>W2V cosine similarity (clipped negative to zero):</h2></p>'
    ret_html    += create_table(tokens1, tokens2, scores)
    ret_html    += '</div>'
    ret_html    += '<div class="floatRight">'
    ret_html    += '<p><h2>One-Hot cosine similarity:</h2></p>'
    ret_html    += create_table(tokens1, tokens2, scores)
    ret_html    += '</div>'
    ret_html    += '</div>'
    ret_html    += '''
    <p><b>Note:</b> Attention scores between the sentences:</p>
    <p>Sentence1: {}</p>
    <p>Sentence2: {}</p>
    </body>
    </html>
    '''.format(sent1, sent2)
    return ret_html

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

