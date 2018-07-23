
# import sys
# print(sys.version)
import platform
python_version = platform.python_version().strip()
print(python_version)
if(python_version.startswith('3')):
    import pickle
else:
    import cPickle as pickle

import re, os, json, subprocess
import random
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from gensim.models.keyedvectors import KeyedVectors
from pprint import pprint
from tqdm import tqdm

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def JsonPredsAppend(preds, data, i, top):
    pref            = "http://www.ncbi.nlm.nih.gov/pubmed/"
    qid             = data['queries'][i]['query_id']
    query           = data['queries'][i]['query_text']
    qdict           = {}
    qdict['body']   = query
    qdict['id']     = qid
    doc_list        = []
    for j in top:
        doc_id      = data['queries'][i]['retrieved_documents'][j]['doc_id']
        doc_list.append(pref + doc_id)
    qdict['documents'] = doc_list
    preds['questions'].append(qdict)

def DumpJson(data, fname):
    with open(fname, 'w') as fw:
        json.dump(data, fw, indent=4)

def tokenize(x):
  return bioclean(x)

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

def load_all_data(dataloc):
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
    idf_pickle_path = '/home/dpappas/IDF_python_v2.pkl'
    idf, max_idf    = load_idfs(idf_pickle_path, words)
    print('loading w2v')
    w2v_bin_path    = '/home/DATA/Biomedical/other/BiomedicalWordEmbeddings/binary/biomedical-vectors-200.bin'
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

my_seed = 1
random.seed(my_seed)
torch.manual_seed(my_seed)

data, docs, tr_data, tr_docs, idf, max_idf, wv = load_all_data('/home/DATA/Biomedical/document_ranking/bioasq_data/')

odir = '/home/dpappas/simplest_posit_drmm/'
if not os.path.exists(odir):
    os.makedirs(odir)

od              = 'sent_posit_drmm_MarginRankingLoss'
k_for_maxpool   = 5
lr              = 0.01

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

def get_map_res(fgold, femit):
    trec_eval_res   = subprocess.Popen(['python', '/home/DATA/Biomedical/document_ranking/eval/run_eval.py', fgold, femit], stdout=subprocess.PIPE, shell=False)
    (out, err)      = trec_eval_res.communicate()
    lines           = out.decode("utf-8").split('\n')
    map_res         = [l for l in lines if (l.startswith('map '))][0].split('\t')
    map_res         = float(map_res[-1])
    return map_res

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, k_for_maxpool, embedding_dim):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.embedding_dim                          = embedding_dim
        # k is for the average k pooling
        self.k                                      = k_for_maxpool
        # Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.trigram_conv                           = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True).double()
        self.trigram_conv_activation                = torch.nn.LeakyReLU()
        #
        self.q_weights_mlp                          = nn.Linear(self.embedding_dim+1, 1, bias=False).double()
        self.linear_per_q1                          = nn.Linear(6, 8, bias=False).double()
        self.linear_per_q2                          = nn.Linear(8, 1, bias=False).double()
        self.out_layer                              = nn.Linear(5, 1, bias=False).double()
        self.my_relu1                               = torch.nn.LeakyReLU()
        self.margin_loss                            = nn.MarginRankingLoss(margin=1.0)
    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta      = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos
    def apply_convolution(self, the_input, the_filters, activation):
        conv_res    = the_filters(the_input.transpose(0,1).unsqueeze(0))
        if(activation is not None):
            conv_res = activation(conv_res)
        pad         = the_filters.padding[0]
        ind_from    = int(np.floor(pad/2.0))
        ind_to      = ind_from + the_input.size(0)
        conv_res    = conv_res[:, :, ind_from:ind_to]
        conv_res    = conv_res.transpose(1, 2)
        conv_res    = conv_res + the_input
        return conv_res.squeeze(0)
    def my_cosine_sim(self,A,B):
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
    def do_for_one_doc(self, doc_embeds, question_embeds, q_conv_res_trigram, q_weights, af):
        sim_insensitive_d               = self.my_cosine_sim(question_embeds, doc_embeds).squeeze(0)
        sim_oh_d                        = (sim_insensitive_d >= 1 - 1e-3).double()
        d_conv_trigram                  = self.apply_convolution(doc_embeds,     self.trigram_conv, self.trigram_conv_activation)
        sim_sensitive_d_trigram         = self.my_cosine_sim(q_conv_res_trigram, d_conv_trigram).squeeze(0)
        sim_insensitive_pooled_d        = self.pooling_method(sim_insensitive_d)
        sim_sensitive_pooled_d_trigram  = self.pooling_method(sim_sensitive_d_trigram)
        sim_oh_pooled_d                 = self.pooling_method(sim_oh_d)
        doc_emit                        = self.get_output([sim_oh_pooled_d, sim_insensitive_pooled_d, sim_sensitive_pooled_d_trigram], q_weights)
        add_feats                       = torch.cat([af, doc_emit.unsqueeze(-1)])
        out                             = self.out_layer(add_feats)
        return out
    def emit_one(self, doc_embeds, question_embeds, q_idfs, gaf):
        q_idfs                          = autograd.Variable(torch.DoubleTensor(q_idfs),          requires_grad=False)
        doc_embeds                      = autograd.Variable(torch.DoubleTensor(doc_embeds),      requires_grad=False)
        question_embeds                 = autograd.Variable(torch.DoubleTensor(question_embeds), requires_grad=False)
        gaf                             = autograd.Variable(torch.DoubleTensor(gaf),             requires_grad=False)
        #
        q_conv_res_trigram              = self.apply_convolution(question_embeds, self.trigram_conv, self.trigram_conv_activation)
        #
        q_weights                       = torch.cat([question_embeds, q_idfs], -1)
        q_weights                       = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights                       = F.softmax(q_weights, dim=-1)
        #
        good_out                        = self.do_for_one_doc(doc_embeds, question_embeds, q_conv_res_trigram, q_weights, gaf)
        return good_out
    def UpdateBatch(self, loss):
        if len(loss) > 0:
            optimizer.zero_grad()
            sum_loss = sum(loss)
            sum_loss.backward()
            optimizer.step()

print('Compiling model...')
logger.info('Compiling model...')
model               = Sent_Posit_Drmm_Modeler(k_for_maxpool=k_for_maxpool, embedding_dim=200)
optimizer           = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
b_size              = 32
max_dev_map         = 0.0
max_epochs          = 30
for epoch in range(max_epochs):
    num_docs, relevant, returned, brelevant, breturned, loss = 0.0, 0.0, 0.0, 0.0, 0.0, []
    train_examples  = GetTrainData(tr_data, 1)
    random.shuffle(train_examples)
    for ex in train_examples:
        i                   = ex[0]
        qtext               = tr_data['queries'][i]['query_text']
        words, _            = get_words(qtext)
        words, qvecs        = get_embeds(words, wv)
        q_idfs              = np.array([[idf_val(qw)] for qw in words], 'float64')
        pos, neg, best_neg  = [], [], -1000000.0
        print(qvecs.shape)
        print(q_idfs.shape)
        for j in ex[1]:
            is_rel          = tr_data['queries'][i]['retrieved_documents'][j]['is_relevant']
            doc_id          = tr_data['queries'][i]['retrieved_documents'][j]['doc_id']
            dtext           = (tr_docs[doc_id]['title'] + ' <title> ' + tr_docs[doc_id]['abstractText'])
            words           = get_words(dtext)
            words, dvecs    = get_embeds(words, wv)
            bm25            = (tr_data['queries'][i]['retrieved_documents'][j]['norm_bm25_score'])
            escores         = GetScores(qtext, dtext, bm25)
            score           = model.emit_one(dvecs, qvecs, q_idfs, escores)
            if is_rel:
                pos.append(score)
            else:
                neg.append(score)
                if score.cpu().item() > best_neg:
                    best_neg = score.cpu().item()
        if pos[0].cpu().item() > best_neg:
            relevant    += 1
            brelevant   += 1
        returned        += 1
        breturned       += 1
        num_docs        += 1
        if len(pos) > 0 and len(neg) > 0:
            loss.append(model.my_hinge_loss(torch.cat(pos), torch.cat(neg), margin=1.0))
        if num_docs % b_size == 0 or num_docs == len(train_examples):
            model.UpdateBatch(loss)
            loss = []
        if num_docs % b_size == 0:
            print('Epoch {}, Instances {}, Cumulative Acc {}, Sub-epoch Acc {}'.format(epoch, num_docs, (float(relevant)/float(returned)), (float(brelevant)/float(breturned))))
            brelevant, breturned = 0, 0
    print('End of epoch {}, Total train docs {} Train Acc {}'.format(epoch, num_docs, (float(relevant)/float(returned))))
    print('Saving model')
    save_checkpoint(epoch, model, max_dev_map, optimizer)
    print('Model saved')
    #
    print('Making Dev preds')
    json_preds, json_preds['questions'], num_docs = {}, [], 0
    for i in range(len(data['queries'])):
        num_docs    += 1
        qtext       = data['queries'][i]['query_text']
        words, _    = get_words(qtext)
        qvecs       = get_embeds(words, wv)
        q_idfs      = np.array([[idf_val(qw)] for qw in words], 'float64')
        rel_scores, rel_scores_sum, sim_matrix = {},{},{}
        for j in range(len(data['queries'][i]['retrieved_documents'])):
            doc_id          = data['queries'][i]['retrieved_documents'][j]['doc_id']
            dtext           = docs[doc_id]['title'] + ' <title> ' + docs[doc_id]['abstractText']
            words, _        = get_words(dtext)
            dvecs           = get_embeds(words, wv)
            bm25            = tr_data['queries'][i]['retrieved_documents'][j]['norm_bm25_score']
            escores         = GetScores(qtext, dtext, bm25)
            score           = model.emit_one(dvecs, qvecs, q_idfs, escores)
            rel_scores[j]   = score.cpu().item()
        top = heapq.nlargest(100, rel_scores, key=rel_scores.get)
        JsonPredsAppend(json_preds, data, i, top)
    DumpJson(json_preds, odir + 'elk_relevant_abs_posit_drmm_lists_dev.json')
    print('Done')

