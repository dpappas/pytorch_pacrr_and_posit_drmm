import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# import sys
# print(sys.version)
import platform
import pprint

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
import logging
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
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def first_alpha_is_upper(sent):
    specials = [
        '__EU__','__SU__','__EMS__','__SMS__','__SI__',
        '__ESB','__SSB__','__EB__','__SB__','__EI__',
        '__EA__','__SA__','__SQ__','__EQ__','__EXTLINK',
        '__XREF','__URI', '__EMAIL','__ARRAY','__TABLE',
        '__FIG','__AWID','__FUNDS'
    ]
    for special in specials:
        sent = sent.replace(special,'')
    for c in sent:
        if(c.isalpha()):
            if(c.isupper()):
                return True
            else:
                return False
    return False

def ends_with_special(sent):
    sent = sent.lower()
    ind = [item.end() for item in re.finditer('[\W\s]sp.|[\W\s]nos.|[\W\s]figs.|[\W\s]sp.[\W\s]no.|[\W\s][vols.|[\W\s]cv.|[\W\s]fig.|[\W\s]e.g.|[\W\s]et[\W\s]al.|[\W\s]i.e.|[\W\s]p.p.m.|[\W\s]cf.|[\W\s]n.a.', sent)]
    if(len(ind)==0):
        return False
    else:
        ind = max(ind)
        if (len(sent) == ind):
            return True
        else:
            return False

def split_sentences(text):
    sents = [l.strip() for l in sent_tokenize(text)]
    ret = []
    i = 0
    while (i < len(sents)):
        sent = sents[i]
        while (
            ((i + 1) < len(sents)) and
            (
                ends_with_special(sent) or
                not first_alpha_is_upper(sents[i+1].strip())
                # sent[-5:].count('.') > 1       or
                # sents[i+1][:10].count('.')>1   or
                # len(sent.split()) < 2          or
                # len(sents[i+1].split()) < 2
            )
        ):
            sent += ' ' + sents[i + 1]
            i += 1
        ret.append(sent.replace('\n',' ').strip())
        i += 1
    return ret

def get_sents(ntext):
    if(len(ntext.strip())>0):
        sents = []
        for subtext in ntext.split('\n'):
            subtext = re.sub( '\s+', ' ', subtext.replace('\n',' ') ).strip()
            if (len(subtext) > 0):
                ss = split_sentences(subtext)
                sents.extend([ s for s in ss if(len(s.strip())>0)])
        if(len(sents[-1]) == 0 ):
            sents = sents[:-1]
        return sents
    else:
        return []

def RemoveTrainLargeYears(data, doc_text):
  for i in range(len(data['queries'])):
    hyear = 1900
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      if data['queries'][i]['retrieved_documents'][j]['is_relevant']:
        doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
        year = doc_text[doc_id]['publicationDate'].split('-')[0]
        if year[:1] == '1' or year[:1] == '2':
          if int(year) > hyear:
            hyear = int(year)
    j = 0
    while True:
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      year = doc_text[doc_id]['publicationDate'].split('-')[0]
      if (year[:1] == '1' or year[:1] == '2') and int(year) > hyear:
        del data['queries'][i]['retrieved_documents'][j]
      else:
        j += 1
      if j == len(data['queries'][i]['retrieved_documents']):
        break
  return data

def RemoveBadYears(data, doc_text, train):
  for i in range(len(data['queries'])):
    j = 0
    while True:
      doc_id    = data['queries'][i]['retrieved_documents'][j]['doc_id']
      year      = doc_text[doc_id]['publicationDate'].split('-')[0]
      ##########################
      # Skip 2017/2018 docs always. Skip 2016 docs for training.
      # Need to change for final model - 2017 should be a train year only.
      # Use only for testing.
      if year == '2017' or year == '2018' or (train and year == '2016'):
      #if year == '2018' or (train and year == '2017'):
        del data['queries'][i]['retrieved_documents'][j]
      else:
        j += 1
      ##########################
      if j == len(data['queries'][i]['retrieved_documents']):
        break
  return data

def print_params(model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(model)
    print(40 * '=')
    # logger.info(40 * '=')
    # logger.info(model)
    # logger.info(40 * '=')
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
    # logger.info(40 * '=')
    # logger.info('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    # logger.info(40 * '=')

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
    # logger.info('Loading IDF tables')
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
    # logger.info('Loaded idf tables with max idf {}'.format(max_idf))
    return ret, max_idf

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

def GetWords(data, doc_text, words):
  for i in range(len(data['queries'])):
    qwds = tokenize(data['queries'][i]['query_text'])
    for w in qwds:
      words[w] = 1
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      dtext = (
              doc_text[doc_id]['title'] + ' <title> ' + doc_text[doc_id]['abstractText'] +
              ' '.join(get_the_mesh(doc_text[doc_id]))
      )
      dwds = tokenize(dtext)
      for w in dwds:
        words[w] = 1

def load_all_data(dataloc, w2v_bin_path, idf_pickle_path):
    print('loading pickle data')
    #
    with open(dataloc+'BioASQ-trainingDataset6b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict( (q['id'], q) for q in bioasq6_data['questions'] )
    # logger.info('loading pickle data')
    with open(dataloc + 'bioasq_bm25_top100.test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.test.pkl', 'rb') as f:
        test_docs = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.dev.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.dev.pkl', 'rb') as f:
        dev_docs = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.train.pkl', 'rb') as f:
        train_docs = pickle.load(f)
    print('loading words')
    #
    train_data  = RemoveBadYears(train_data, train_docs, True)
    train_data  = RemoveTrainLargeYears(train_data, train_docs)
    dev_data    = RemoveBadYears(dev_data, dev_docs, False)
    test_data   = RemoveBadYears(test_data, test_docs, False)
    #
    words           = {}
    GetWords(train_data, train_docs, words)
    GetWords(dev_data,   dev_docs,   words)
    GetWords(test_data,  test_docs,  words)
    # mgmx
    print('loading idfs')
    idf, max_idf    = load_idfs(idf_pickle_path, words)
    print('loading w2v')
    wv              = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    wv              = dict([(word, wv[word]) for word in wv.vocab.keys() if(word in words)])
    return test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, wv, bioasq6_data

def train_data_yielder():
    for dato in tqdm(train_data['queries']):
        quest       = dato['query_text']
        bm25s       = {t['doc_id']: t['norm_bm25_score'] for t in dato[u'retrieved_documents']}
        ret_pmids   = [t[u'doc_id'] for t in dato[u'retrieved_documents']]
        good_pmids  = [t for t in ret_pmids if t in dato[u'relevant_documents']]
        bad_pmids   = [t for t in ret_pmids if t not in dato[u'relevant_documents']]
        #
        quest_tokens, quest_embeds = get_embeds(tokenize(quest), wv)
        q_idfs      = np.array([[idf_val(qw)] for qw in quest_tokens], 'float64')
        #
        if(len(bad_pmids)>0):
            for gid in good_pmids:
                bid                         = random.choice(bad_pmids)
                good_text                   = train_docs[gid]['title'] + ' <title> ' + train_docs[gid]['abstractText']
                good_tokens, good_embeds    = get_embeds(tokenize(good_text), wv)
                bad_text                    = train_docs[bid]['title'] + ' <title> ' + train_docs[bid]['abstractText']
                bad_tokens, bad_embeds      = get_embeds(tokenize(bad_text), wv)
                good_escores                = GetScores(quest, good_text, bm25s[gid])
                bad_escores                 = GetScores(quest, bad_text,  bm25s[bid])
                yield(good_embeds, bad_embeds, quest_embeds, q_idfs, good_escores, bad_escores)

def train_data_step1():
    # bioasq6_data
    ret = []
    for dato in tqdm(train_data['queries']):
        quest       = dato['query_text']
        quest_id    = dato['query_id']
        bm25s       = {t['doc_id']: t['norm_bm25_score'] for t in dato[u'retrieved_documents']}
        ret_pmids   = [t[u'doc_id'] for t in dato[u'retrieved_documents']]
        good_pmids  = [t for t in ret_pmids if t in dato[u'relevant_documents']]
        bad_pmids   = [t for t in ret_pmids if t not in dato[u'relevant_documents']]
        if(len(bad_pmids)>0):
            for gid in good_pmids:
                bid = random.choice(bad_pmids)
                ret.append((quest, quest_id, gid, bid, bm25s[gid], bm25s[bid]))
    print('')
    return ret

def get_snips(quest_id, gid):
    good_snips = []
    if('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            if (sn['document'].endswith(gid)):
                good_snips.extend(get_sents(sn['text']))
    return good_snips

def get_sent_tags(good_sents, good_snips):
    sent_tags = []
    for sent in good_sents:
        sent_tags.append(int((sent in good_snips) or any([s in sent for s in good_snips])))
    return sent_tags

def get_the_mesh(the_doc):
    good_mesh = []
    for t in the_doc['meshHeadingsList']:
        t = t.split(':', 1)
        t = t[1].strip()
        t = t.lower()
        good_mesh.append(t)
    good_mesh = sorted(good_mesh)
    good_mesh = ['mgmx'] + good_mesh
    good_mesh = ' # '.join(good_mesh)
    good_mesh = good_mesh.split()
    return good_mesh

def train_data_step2(train_instances):
    for quest, quest_id, gid, bid, bm25s_gid, bm25s_bid in train_instances:
        quest_tokens, quest_embeds              = get_embeds(tokenize(quest), wv)
        q_idfs                                  = np.array([[idf_val(qw)] for qw in quest_tokens], 'float')
        #
        good_mesh                               = get_the_mesh(train_docs[gid])
        good_doc_text                           = train_docs[gid]['title'] + train_docs[gid]['abstractText']
        good_doc_af                             = GetScores(quest, good_doc_text, bm25s_gid)
        good_sents_title                        = get_sents(train_docs[gid]['title'])
        good_sents_abs                          = get_sents(train_docs[gid]['abstractText'])
        #
        good_sents                              = good_sents_title + good_sents_abs
        #
        good_snips                              = get_snips(quest_id, gid)
        good_snips                              = [' '.join(bioclean(sn)) for sn in good_snips]
        #
        good_sents_embeds, good_sents_escores, good_sent_tags = [], [], []
        for good_text in good_sents:
            good_tokens, good_embeds            = get_embeds(tokenize(good_text), wv)
            good_escores                        = GetScores(quest, good_text, bm25s_gid)[:-1]
            if(len(good_embeds)>0):
                good_sents_embeds.append(good_embeds)
                good_sents_escores.append(good_escores)
                tt = ' '.join(bioclean(good_text))
                good_sent_tags.append(int((tt in good_snips) or any([s in tt for s in good_snips])))
        #
        bad_mesh                                = get_the_mesh(train_docs[bid])
        bad_doc_text                            = train_docs[bid]['title'] + train_docs[bid]['abstractText']
        bad_doc_af                              = GetScores(quest, bad_doc_text, bm25s_bid)
        bad_sents                               = get_sents(train_docs[bid]['title']) + get_sents(train_docs[bid]['abstractText'])
        #
        bad_sent_tags                           = len(bad_sents) * [0]
        #
        bad_sents_embeds, bad_sents_escores     = [], []
        for bad_text in bad_sents:
            bad_tokens, bad_embeds              = get_embeds(tokenize(bad_text), wv)
            bad_escores                         = GetScores(quest, bad_text, bm25s_bid)[:-1]
            if(len(bad_embeds)>0):
                bad_sents_embeds.append(bad_embeds)
                bad_sents_escores.append(bad_escores)
        if(sum(good_sent_tags)>0):
            bmt, bad_mesh_embeds    = get_embeds(bad_mesh, wv)
            gmt, good_mesh_embeds   = get_embeds(good_mesh, wv)
            yield (
                good_sents_embeds,  bad_sents_embeds,   quest_embeds,       q_idfs,
                good_sents_escores, bad_sents_escores,  good_doc_af,        bad_doc_af,
                good_sent_tags,     bad_sent_tags,      good_mesh_embeds,   bad_mesh_embeds
            )

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_sim_info(sent, snippets):
    sent            = bioclean(sent)
    similarities    = [similar(sent, s) for s in snippets]
    max_sim         = max(similarities)
    index_of_max    = similarities.index(max_sim)
    return max_sim, index_of_max, snippets[index_of_max]

w2v_bin_path    = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
idf_pickle_path = '/home/dpappas/for_ryan/fordp/idf.pkl'
dataloc         = '/home/dpappas/for_ryan/'
eval_path       = '/home/dpappas/for_ryan/eval/run_eval.py'

test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, wv, bioasq6_data = load_all_data(
    dataloc         = dataloc,
    w2v_bin_path    = w2v_bin_path,
    idf_pickle_path = idf_pickle_path
)

train_instances = train_data_step1()
# train_instances = train_instances[:len(train_instances)/2]
epoch_aver_cost, epoch_aver_acc = 0., 0.
random.shuffle(train_instances)
for instance in train_data_step2(train_instances):


