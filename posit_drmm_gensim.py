
# import sys
# print(sys.version)
import platform
python_version = platform.python_version().strip()
print(python_version)
if(python_version.startswith('3')):
    import pickle
else:
    import cPickle as pickle

import os
import re
import json
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pprint import pprint
import torch.autograd as autograd
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

my_seed = random.randint(0,2000000)
random.seed(my_seed)
torch.manual_seed(my_seed)

w2v_bin_path    = '/home/dpappas/for_ryan/biomedical-vectors-200.bin'
idf_pickle_path = '/home/dpappas/for_ryan/IDF_python_v2.pkl'
dataloc         = '/home/dpappas/for_ryan/'

odir            = '/home/dpappas/posit_drmm_gensim/'
if not os.path.exists(odir):
    os.makedirs(odir)

od              = 'sent_posit_drmm_MarginRankingLoss'
k_for_maxpool   = 5
embedding_dim   = 200
lr              = 0.01
bsize           = 32

import logging
logger      = logging.getLogger(od)
hdlr        = logging.FileHandler(odir+'model.log')
formatter   = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

def print_params(model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(model)
    print(40 * '=')
    logger.info(40 * '=')
    logger.info(model)
    logger.info(40 * '=')
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
    logger.info(40 * '=')
    logger.info('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    logger.info(40 * '=')

def dummy_test():
    doc1_embeds         = np.random.rand(40, 200)
    doc2_embeds         = np.random.rand(30, 200)
    question_embeds     = np.random.rand(20, 200)
    q_idfs              = np.random.rand(20, 1)
    gaf                 = np.random.rand(4)
    baf                 = np.random.rand(4)
    for epoch in range(200):
        optimizer.zero_grad()
        cost_, doc1_emit_, doc2_emit_ = model(
            doc1_embeds     = doc1_embeds,
            doc2_embeds     = doc2_embeds,
            question_embeds = question_embeds,
            q_idfs          = q_idfs,
            gaf             = gaf,
            baf             = baf
        )
        cost_.backward()
        optimizer.step()
        the_cost = cost_.cpu().item()
        print(the_cost, float(doc1_emit_), float(doc2_emit_))
    print(20 * '-')

def compute_the_cost(costs, back_prop=True):
    cost_ = torch.stack(costs)
    cost_ = cost_.sum() / (1.0 * cost_.size(0))
    if(back_prop):
        cost_.backward()
        optimizer.step()
        optimizer.zero_grad()
    the_cost = cost_.cpu().item()
    return the_cost

def save_checkpoint(epoch, model, max_dev_map, optimizer, filename='checkpoint.pth.tar'):
    '''
    :param state:       the stete of the pytorch mode
    :param filename:    the name of the file in which we will store the model.
    :return:            Nothing. It just saves the model.
    '''
    state = {
        'epoch':            epoch,
        'state_dict':       model.state_dict(),
        'best_valid_score': max_dev_map,
        'optimizer':        optimizer.state_dict(),
    }
    torch.save(state, filename)

def train_one(train_instances):
    costs   = []
    optimizer.zero_grad()
    instance_metr, average_total_loss, average_task_loss, average_reg_loss = 0.0, 0.0, 0.0, 0.0
    for good_sents_inds, good_all_sims, bad_sents_inds, bad_all_sims, quest_inds, gaf, baf in train_instances:
        instance_cost, doc1_emit, doc2_emit, loss1, loss2 = model(good_sents_inds, bad_sents_inds, quest_inds, good_all_sims, bad_all_sims, gaf, baf)
        #
        average_total_loss  += instance_cost.cpu().item()
        average_task_loss   += loss1.cpu().item()
        average_reg_loss    += loss2.cpu().item()
        #
        instance_metr       += 1
        costs.append(instance_cost)
        if(len(costs) == bsize):
            batch_loss      = compute_the_cost(costs, True)
            costs = []
            print('train epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch,instance_metr,average_total_loss/(1.*instance_metr),average_task_loss/(1.*instance_metr),average_reg_loss/(1.*instance_metr)))
            logger.info('train epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch,instance_metr,average_total_loss/(1.*instance_metr),average_task_loss/(1.*instance_metr),average_reg_loss/(1.*instance_metr)))
    if(len(costs)>0):
        batch_loss = compute_the_cost(costs, True)
        print('train epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch, instance_metr, average_total_loss/(1.*instance_metr), average_task_loss/(1.*instance_metr), average_reg_loss/(1.*instance_metr)))
        logger.info('train epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch, instance_metr, average_total_loss/(1.*instance_metr), average_task_loss/(1.*instance_metr), average_reg_loss/(1.*instance_metr)))
    return average_task_loss / instance_metr

def dev_one(dev_instances):
    optimizer.zero_grad()
    instance_metr, average_total_loss, average_task_loss, average_reg_loss = 0.0, 0.0, 0.0, 0.0
    for good_sents_inds, good_all_sims, bad_sents_inds, bad_all_sims, quest_inds, gaf, baf in dev_instances:
        instance_cost, doc1_emit, doc2_emit, loss1, loss2 = model(good_sents_inds, bad_sents_inds, quest_inds, good_all_sims, bad_all_sims, gaf, baf)
        average_total_loss  += instance_cost.cpu().item()
        average_task_loss   += loss1.cpu().item()
        average_reg_loss    += loss2.cpu().item()
        instance_metr       += 1
    print('dev epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch, instance_metr, average_total_loss/(1.*instance_metr), average_task_loss/(1.*instance_metr), average_reg_loss/(1.*instance_metr)))
    logger.info('dev epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch, instance_metr, average_total_loss/(1.*instance_metr), average_task_loss/(1.*instance_metr), average_reg_loss/(1.*instance_metr)))
    return average_task_loss / instance_metr

def get_map_res(fgold, femit):
    trec_eval_res   = subprocess.Popen(['python', '/home/DATA/Biomedical/document_ranking/eval/run_eval.py', fgold, femit], stdout=subprocess.PIPE, shell=False)
    (out, err)      = trec_eval_res.communicate()
    lines           = out.decode("utf-8").split('\n')
    map_res         = [l for l in lines if (l.startswith('map '))][0].split('\t')
    map_res         = float(map_res[-1])
    return map_res

def tokenize(x):
  return bioclean(x)

def idf_val(w):
    if w in idf:
        return idf[w]
    return max_idf

def get_words(s):
    sl  = tokenize(s)
    sl  = [s for s in sl]
    sl2 = [s for s in sl if idf_val(s) >= 2.0]
    return sl, sl2

def get_embeds(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if(tok in wv):
            ret1.append(tok)
            ret2.append(wv[tok])
    return ret1, np.array(ret2, 'float64')

def load_idfs(idf_path, words):
    print('Loading IDF tables')
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
    print('Loaded idf tables with max idf %f' % max_idf)
    return ret, max_idf

def GetWords(data, doc_text, words):
  for i in range(len(data['queries'])):
    qwds = tokenize(data['queries'][i]['query_text'])
    for w in qwds:
      words[w] = 1
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      dtext = (doc_text[doc_id]['title'] + ' <title> ' +
               doc_text[doc_id]['abstractText'])
      dwds = tokenize(dtext)
      for w in dwds:
        words[w] = 1

def load_all_data(dataloc, w2v_bin_path, idf_pickle_path):
    print('loading pickle data')
    with open(dataloc + 'bioasq_bm25_top100.dev.pkl', 'rb') as f:
      data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.dev.pkl', 'rb') as f:
      docs = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.train.pkl', 'rb') as f:
      tr_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.train.pkl', 'rb') as f:
      tr_docs = pickle.load(f)
    print('loading words')
    words = {}
    GetWords(tr_data, tr_docs, words)
    GetWords(data, docs, words)
    print('loading idfs')
    idf, max_idf    = load_idfs(idf_pickle_path, words)
    print('loading w2v')
    wv              = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    return data, docs, tr_data, tr_docs, idf, max_idf, wv

def GetTrainData(data, max_neg=1):
  train_data = []
  for i in range(len(data['queries'])):
    pos, neg = [], []
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      is_rel = data['queries'][i]['retrieved_documents'][j]['is_relevant']
      if is_rel:
        pos.append(j)
      else:
        neg.append(j)
    if len(pos) > 0 and len(neg) > 0:
      for p in pos:
        neg_ex = []
        if len(neg) <= max_neg:
          neg_ex = neg
        else:
          used = {}
          while len(neg_ex) < max_neg:
            n = random.randint(0, len(neg)-1)
            if n not in used:
              neg_ex.append(neg[n])
              used[n] = 1
        inst = [i, [p] + neg_ex]
        train_data.append(inst)
  return train_data

def query_doc_overlap(qwords, dwords):
    # % Query words in doc.
    qwords_in_doc = 0
    idf_qwords_in_doc = 0.0
    idf_qwords = 0.0
    for qword in uwords(qwords):
      idf_qwords += idf_val(qword)
      for dword in uwords(dwords):
        if qword == dword:
          idf_qwords_in_doc += idf_val(qword)
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
      idf_bigrams += idf_val(wrds[0]) * idf_val(wrds[1])
      for dword in ubigrams(dwords):
        if qword == dword:
          qwords_bigrams_in_doc += 1
          idf_qwords_bigrams_in_doc += (idf_val(wrds[0]) * idf_val(wrds[1]))
          break
    if len(qwords) <= 0:
      qwords_bigrams_in_doc_val = 0.0
    else:
      qwords_bigrams_in_doc_val = (float(qwords_bigrams_in_doc) / float(len(ubigrams(qwords))))
    if idf_bigrams <= 0.0:
      idf_qwords_bigrams_in_doc_val = 0.0
    else:
      idf_qwords_bigrams_in_doc_val = (float(idf_qwords_bigrams_in_doc) / float(idf_bigrams))
    return [qwords_in_doc_val,
            qwords_bigrams_in_doc_val,
            idf_qwords_in_doc_val,
            idf_qwords_bigrams_in_doc_val]

def GetScores(qtext, dtext, bm25):
    qwords, qw2 = get_words(qtext)
    dwords, dw2 = get_words(dtext)
    qd1         = query_doc_overlap(qwords, dwords)
    bm25        = [bm25]
    return qd1[0:3] + bm25

def handle_tr_quest(i):
    qtext           = tr_data['queries'][i]['query_text']
    words, _        = get_words(qtext)
    words, qvecs    = get_embeds(words, wv)
    q_idfs          = np.array([[idf_val(qw)] for qw in words], 'float64')
    return qtext, qvecs, q_idfs

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim, k_for_maxpool):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k                                      = k_for_maxpool         # k is for the average k pooling
        #
        self.embedding_dim                          = embedding_dim
        self.trigram_conv                           = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation                = torch.nn.LeakyReLU()
        #
        self.q_weights_mlp                          = nn.Linear(self.embedding_dim+1, 1, bias=False)
        self.linear_per_q1                          = nn.Linear(6, 8, bias=False)
        self.linear_per_q2                          = nn.Linear(8, 1, bias=False)
        self.my_relu1                               = torch.nn.LeakyReLU()
        self.margin_loss                            = nn.MarginRankingLoss(margin=1.0)
        self.out_layer                              = nn.Linear(5, 1, bias=False)
    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta      = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos
    def apply_convolution(self, the_input, the_filters, activation):
        conv_res        = the_filters(the_input.transpose(0,1).unsqueeze(0))
        if(activation is not None):
            conv_res    = activation(conv_res)
        pad             = the_filters.padding[0]
        ind_from        = int(np.floor(pad/2.0))
        ind_to          = ind_from + the_input.size(0)
        conv_res        = conv_res[:, :, ind_from:ind_to]
        conv_res        = conv_res.transpose(1, 2)
        conv_res        = conv_res + the_input
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
        sorted_res              = torch.sort(sim_matrix, -1)[0]             # sort the input minimum to maximum
        k_max_pooled            = sorted_res[:,-self.k:]                    # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)        # average these k values
        the_maximum             = k_max_pooled[:, -1]                       # select the maximum value of each instance
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled], dim=-1) # concatenate maximum value and average of k-max values
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
    def fix_input_one(self, doc1_embeds, question_embeds, q_idfs, gaf):
        doc1_embeds     = autograd.Variable(torch.FloatTensor(doc1_embeds),     requires_grad=False)
        question_embeds = autograd.Variable(torch.FloatTensor(question_embeds), requires_grad=False)
        gaf             = autograd.Variable(torch.FloatTensor(gaf),             requires_grad=False)
        q_idfs          = autograd.Variable(torch.FloatTensor(q_idfs),          requires_grad=False)
        return doc1_embeds, question_embeds, q_idfs, gaf
    def fix_input_two(self, doc1_embeds, doc2_embeds, question_embeds, q_idfs, gaf, baf):
        doc1_embeds     = autograd.Variable(torch.FloatTensor(doc1_embeds),     requires_grad=False)
        doc2_embeds     = autograd.Variable(torch.FloatTensor(doc2_embeds),     requires_grad=False)
        question_embeds = autograd.Variable(torch.FloatTensor(question_embeds), requires_grad=False)
        gaf             = autograd.Variable(torch.FloatTensor(gaf),             requires_grad=False)
        baf             = autograd.Variable(torch.FloatTensor(baf),             requires_grad=False)
        q_idfs          = autograd.Variable(torch.FloatTensor(q_idfs),          requires_grad=False)
        return doc1_embeds, doc2_embeds, question_embeds, q_idfs, gaf, baf
    def emit_one(self, doc1_embeds, question_embeds, q_idfs, gaf):
        doc1_embeds, question_embeds, q_idfs, gaf = self.fix_input_one(doc1_embeds, question_embeds, q_idfs, gaf)
        pass
    def forward(self, doc1_embeds, doc2_embeds, question_embeds, q_idfs, gaf, baf):
        doc1_embeds, doc2_embeds, question_embeds, q_idfs, gaf, baf = self.fix_input_two(
            doc1_embeds, doc2_embeds, question_embeds, q_idfs, gaf, baf
        )
        # cosine similarity on pretrained word embeddings
        sim_insensitive_d1              = self.my_cosine_sim(question_embeds, doc1_embeds).squeeze(0)
        sim_insensitive_d2              = self.my_cosine_sim(question_embeds, doc2_embeds).squeeze(0)
        sim_oh_d1                       = (sim_insensitive_d1 > (1-(1e-3))).float()
        sim_oh_d2                       = (sim_insensitive_d2 > (1-(1e-3))).float()
        # 3gram convolution on the embedding matrix
        q_conv_res_trigram              = self.apply_convolution(question_embeds, self.trigram_conv, self.trigram_conv_activation)
        d1_conv_trigram                 = self.apply_convolution(doc1_embeds,     self.trigram_conv, self.trigram_conv_activation)
        d2_conv_trigram                 = self.apply_convolution(doc2_embeds,     self.trigram_conv, self.trigram_conv_activation)
        # cosine similairy on the contextual embeddings
        sim_sensitive_d1_trigram        = self.my_cosine_sim(q_conv_res_trigram, d1_conv_trigram).squeeze(0)
        sim_sensitive_d2_trigram        = self.my_cosine_sim(q_conv_res_trigram, d2_conv_trigram).squeeze(0)
        # pooling 3 * 2 fetures from the similarity matrices for the good doc
        sim_insensitive_pooled_d1       = self.pooling_method(sim_insensitive_d1)
        sim_sensitive_pooled_d1_trigram = self.pooling_method(sim_sensitive_d1_trigram)
        sim_oh_pooled_d1                = self.pooling_method(sim_oh_d1)
        # pooling 3 * 2 fetures from the similarity matrices for the bad doc
        sim_insensitive_pooled_d2       = self.pooling_method(sim_insensitive_d2)
        sim_sensitive_pooled_d2_trigram = self.pooling_method(sim_sensitive_d2_trigram)
        sim_oh_pooled_d2                = self.pooling_method(sim_oh_d2)
        # create the weights for weighted average
        q_weights                       = torch.cat([q_conv_res_trigram, q_idfs], -1)
        q_weights                       = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights                       = F.softmax(q_weights, dim=-1)
        # concatenate and pass through mlps
        doc1_emit                       = self.get_output([sim_oh_pooled_d1, sim_insensitive_pooled_d1, sim_sensitive_pooled_d1_trigram], q_weights)
        doc2_emit                       = self.get_output([sim_oh_pooled_d2, sim_insensitive_pooled_d2, sim_sensitive_pooled_d2_trigram], q_weights)
        # concatenate the mlps' output to the additional features
        good_add_feats                  = torch.cat([gaf, doc1_emit.unsqueeze(-1)])
        bad_add_feats                   = torch.cat([baf, doc2_emit.unsqueeze(-1)])
        # apply output layer
        good_out                        = self.out_layer(good_add_feats)
        bad_out                         = self.out_layer(bad_add_feats)
        # compute the loss
        loss1                           = self.margin_loss(good_out, bad_out, torch.ones(1))
        # loss1                           = self.my_hinge_loss(good_out, bad_out)
        return loss1, good_out, bad_out

print('Compiling model...')
logger.info('Compiling model...')
model       = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool)
params      = model.parameters()
print_params(model)
optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# dummy_test()

data, docs, tr_data, tr_docs, idf, max_idf, wv = load_all_data(dataloc=dataloc, w2v_bin_path=w2v_bin_path, idf_pickle_path=idf_pickle_path)

train_examples = GetTrainData(tr_data, 1)
random.shuffle(train_examples)

for ex in train_examples:
    i = ex[0]
    qtext, qvecs, q_idfs = handle_tr_quest(i)


exit()

# train_all_abs, dev_all_abs, test_all_abs, train_bm25_scores, dev_bm25_scores, test_bm25_scores, t2i = load_data()
#
# data, docs, tr_data, tr_docs, idf, max_idf, wv = load_all_data('/home/DATA/Biomedical/document_ranking/bioasq_data/')
# fgold = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq.dev.json'

# max_dev_map     = 0.0
min_dev_loss    = 10e5
max_epochs      = 30
loopes          = [1, 0, 0]
dev_instances   = list(random_data_yielder(dev_bm25_scores, dev_all_abs, t2i, bsize * 50))
for epoch in range(max_epochs):
    train_instances         = random_data_yielder(train_bm25_scores, train_all_abs, t2i, bsize * 100)
    train_average_loss      = train_one(train_instances)
    dev_average_loss        = dev_one(dev_instances)
    # dev_map                 = get_one_map('dev', dev_bm25_scores, dev_all_abs)
    if(min_dev_loss > dev_average_loss):
        min_dev_loss        = dev_average_loss
        min_loss_epoch      = epoch+1
        test_map            = get_one_map('test', test_bm25_scores, test_all_abs)
        save_checkpoint(epoch, model, dev_average_loss, optimizer, filename=odir+'best_checkpoint.pth.tar')
    print("epoch:{}, train_average_loss:{}, dev_map:{}, test_map:{}".format(epoch+1, train_average_loss, dev_average_loss, test_map))
    print(20 * '-')
    logger.info("epoch:{}, train_average_loss:{}, dev_map:{}, test_map:{}".format(epoch+1, train_average_loss, dev_average_loss, test_map))
    logger.info(20 * '-')


