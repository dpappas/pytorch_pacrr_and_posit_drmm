#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# # import sys
# # print(sys.version)
# import platform
# import pprint
#
# python_version = platform.python_version().strip()
# print(python_version)
# if(python_version.startswith('3')):
#     import pickle
# else:
#     import cPickle as pickle

import  os
import  re
import  json
import  random
import  logging
import  subprocess
import  numpy as np
import  torch
import  torch.nn as nn
import  torch.optim as optim
import  torch.nn.functional as F
from    pprint import pprint
import  cPickle as pickle
import  torch.autograd as autograd
from    tqdm import tqdm
from    gensim.models.keyedvectors import KeyedVectors
from    nltk.tokenize import sent_tokenize
from    difflib import SequenceMatcher

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

def get_map_res(fgold, femit):
    trec_eval_res   = subprocess.Popen(['python', eval_path, fgold, femit], stdout=subprocess.PIPE, shell=False)
    (out, err)      = trec_eval_res.communicate()
    lines           = out.decode("utf-8").split('\n')
    map_res         = [l for l in lines if (l.startswith('map '))][0].split('\t')
    map_res         = float(map_res[-1])
    return map_res

def get_bioasq_res(prefix, data_gold, data_emitted):
    '''
    java -Xmx10G -cp /home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar
    evaluation.EvaluatorTask1b -phaseA -e 5
    /home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/BioASQ-task6bPhaseB-testset1
    ./drmm-experimental_submit.json
    '''
    jar_path = '/home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar'
    #
    fgold    = './{}_gold_bioasq.json'.format(prefix)
    fgold    = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_gold, indent=4, sort_keys=True))
        f.close()
    #
    femit    = './{}_emit_bioasq.json'.format(prefix)
    femit    = os.path.abspath(femit)
    with open(femit, 'w') as f:
        f.write(json.dumps(data_emitted, indent=4, sort_keys=True))
        f.close()
    #
    bioasq_eval_res = subprocess.Popen(
        [
            'java', '-Xmx10G', '-cp', jar_path, 'evaluation.EvaluatorTask1b',
            '-phaseA', '-e', '5', fgold, femit
        ],
        stdout=subprocess.PIPE, shell=False
    )
    (out, err)  = bioasq_eval_res.communicate()
    lines       = out.decode("utf-8").split('\n')
    ret = {}
    for line in lines:
        if('GMAP snippets:' in line):
            ret['GMAP'] = float(line.split()[-1])
        elif('MAP snippets:' in line):
            ret['MAP'] = float(line.split()[-1])
        elif('F1 snippets:' in line):
            ret['F1'] = float(line.split()[-1])
    return ret

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
    logger.info('loading words')
    #
    words           = {}
    GetWords(train_data, train_docs, words)
    GetWords(dev_data,   dev_docs,   words)
    GetWords(test_data,  test_docs,  words)
    # mgmx
    print('loading idfs')
    logger.info('loading idfs')
    idf, max_idf    = load_idfs(idf_pickle_path, words)
    print('loading w2v')
    logger.info('loading w2v')
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
    logger.info('')
    return ret

def get_snips(quest_id, gid):
    good_snips = []
    if('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            if (sn['document'].endswith(gid)):
                good_snips.extend(get_sents(sn['text']))
    return good_snips

def get_gold_snips(quest_id):
    gold_snips                  = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            gold_snips.extend(get_sents(sn['text']))
    return gold_snips

def get_sent_tags(good_sents, good_snips):
    sent_tags = []
    for sent in good_sents:
        sent_tags.append(int((sent in good_snips) or any([s in sent for s in good_snips])))
    return sent_tags

def get_the_mesh(the_doc):
    good_mesh = []
    if('meshHeadingsList' in the_doc):
        for t in the_doc['meshHeadingsList']:
            t = t.split(':', 1)
            t = t[1].strip()
            t = t.lower()
            good_mesh.append(t)
    elif('MeshHeadings' in the_doc):
        for mesh_head_set in the_doc['MeshHeadings']:
            for item in mesh_head_set:
                good_mesh.append(item['text'].strip().lower())
    if('Chemicals' in the_doc):
        for t in the_doc['Chemicals']:
            t = t['NameOfSubstance'].strip().lower()
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

def back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc):
    batch_cost = sum(batch_costs) / float(len(batch_costs))
    batch_cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    batch_aver_cost = batch_cost.cpu().item()
    epoch_aver_cost = sum(epoch_costs) / float(len(epoch_costs))
    batch_aver_acc  = sum(batch_acc) / float(len(batch_acc))
    epoch_aver_acc  = sum(epoch_acc) / float(len(epoch_acc))
    return batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc

def handle_good(retr, quest):
    fpath                       = '/home/dpappas/for_ryan/downloaded/{}.json'.format(retr['doc_id'])
    json_dato                   = json.load(open(fpath))
    if(len(json_dato)>0):
        if('AbstractText' not in json_dato and 'ArticleTitle' not in json_dato):
            return None
        json_dato['ArticleTitle']   = fix_text_for_matching(json_dato['ArticleTitle'])
        json_dato['AbstractText']   = fix_text_for_matching(json_dato['AbstractText'])
        good_doc_text               = json_dato['ArticleTitle'] + ' ' + json_dato['AbstractText']
        good_doc_af                 = GetScores(quest, good_doc_text, retr['norm_bm25_score'])
        good_sents_embeds           = []
        good_sents_escores          = []
        held_out_sents              = []
        for good_text in get_sents(json_dato['ArticleTitle']):
            good_tokens, good_embeds    = get_embeds(tokenize(good_text), wv)
            good_escores                = GetScores(quest, good_text, retr['norm_bm25_score'])[:-1]
            if (len(good_embeds) > 0):
                good_sents_embeds.append(good_embeds)
                good_sents_escores.append(good_escores)
                held_out_sents.append(
                    (
                        'title',
                        good_text,
                        json_dato['ArticleTitle'].index(good_text),
                        json_dato['ArticleTitle'].index(good_text)+len(good_text)
                    )
                )
        #
        for good_text in get_sents(json_dato['AbstractText']):
            good_tokens, good_embeds    = get_embeds(tokenize(good_text), wv)
            good_escores                = GetScores(quest, good_text, retr['norm_bm25_score'])[:-1]
            if (len(good_embeds) > 0):
                good_sents_embeds.append(good_embeds)
                good_sents_escores.append(good_escores)
                good_text = fix_text_for_matching(good_text)
                try:
                    held_out_sents.append(
                        (
                            'abstract',
                            good_text,
                            json_dato['AbstractText'].index(good_text),
                            json_dato['AbstractText'].index(good_text)+len(good_text)
                        )
                    )
                except:
                    print(retr['doc_id'])
                    print(good_text)
                    print(20 * '-')
                    print(json_dato['AbstractText'])
                    print(20 * '=')
        good_mesh               = get_the_mesh(json_dato)
        gmt, good_mesh_embeds   = get_embeds(good_mesh, wv)
        return good_sents_embeds, good_sents_escores, good_doc_af, good_mesh_embeds, held_out_sents
    else:
        return None

def snip_is_relevant(one_sent, gold_snips):
    return any(
        [
            (one_sent in gold_snip) or (gold_snip in one_sent)
            for gold_snip in gold_snips
        ]
    )

def print_snip_res(dato, emitss, held_out_sents, gold_snips, retr):
        print dato['query_text']
        for i in range(len(emitss)):
            print(
                '{}\t{}\t{}\t{}'.format(
                    snip_is_relevant(held_out_sents[i], gold_snips),
                    emitss[i],
                    "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(retr['doc_id']),
                    held_out_sents[i]
                )
            )

def get_pseudo_retrieved(dato):
    some_ids                    = [
        item['document'].split('/')[-1].strip()
        for item in bioasq6_data[dato['query_id']]['snippets']
    ]
    pseudo_retrieved            = [
        {
            'bm25_score'        : 7.76,
            'doc_id'            : id,
            'is_relevant'       : True,
            'norm_bm25_score'   : 3.85
        }
        for id in set(some_ids)
    ]
    return pseudo_retrieved

def prep_bioasq_subm_dato(extracted_snippets, dato, doc_res):
    snipis = []
    for sn in extracted_snippets:
        snipis.append(
            {
                "beginSection"          : sn[3][0],
                "endSection"            : sn[3][0],
                "offsetInBeginSection"  : sn[3][2],
                "offsetInEndSection"    : sn[3][3],
                "text"                  : sn[3][1],
                "document"              : sn[2]
            }
        )
    bioasq_subm_dato            = {
        'body'      : dato['query_text'],
        'documents' : doc_res,
        'id'        : dato['query_id'],
        'snippets'  : snipis
    }
    return bioasq_subm_dato

def fix_text_for_matching(the_text):
    the_text = the_text.replace('or intramuscular interferon beta-1a (30 ìg) once a week.', 'or intramuscular interferon beta-1a (30 μg) once a week.')
    the_text = the_text.replace('(bromocriptine, lisuride, pergolide etc.),', '(bromocriptine, lisuride, pergolide etc. ),')
    the_text = the_text.replace('AVAILABILITY AND IMPLEMENTATION:', '\n')
    the_text = the_text.replace('METHODOLOGY/PRINCIPAL FINDINGS:',  '\n')
    the_text = the_text.replace('BACKGROUND AND OBJECTIVES:',       '\n')
    the_text = the_text.replace('DESIGN AND METHODS:',              '\n')
    the_text = the_text.replace('METHODS AND ANALYSIS:',            '\n')
    the_text = the_text.replace('BACKGROUND AND S:',                '\n')
    the_text = the_text.replace('BACKGROUND & AIMS:',               '\n')
    the_text = the_text.replace('STUDY OBJECTIVES:',                '\n')
    the_text = the_text.replace('RECENT FINDINGS:',                 '\n')
    the_text = the_text.replace('AVAILABILITY:',                    '\n')
    the_text = the_text.replace('INTRODUCTION:',                    '\n')
    the_text = the_text.replace('INTERPRETATION:',                  '\n')
    the_text = the_text.replace('BACKGROUND/ S:',                   '\n')
    the_text = the_text.replace('PURPOSE:',     '\n')
    the_text = the_text.replace('BACKGROUND:',  '\n')
    the_text = the_text.replace('METHODS:',     '\n')
    the_text = the_text.replace('CONTEXT:',     '\n')
    the_text = the_text.replace('OBJECTIVE:',   '\n')
    the_text = the_text.replace('OBJECTIVES:',  '\n')
    the_text = the_text.replace('RESULTS:',     '\n')
    the_text = the_text.replace('SUMMARY:',     '\n')
    the_text = the_text.replace('CONCLUSION:',  '\n')
    the_text = the_text.replace('CONCLUSIONS:', '\n')
    the_text = the_text.replace('UNLABELLED:',  '\n')
    the_text = the_text.replace('PATIENTS:',    '\n')
    the_text = the_text.replace('DISCUSSION:',  '\n')
    the_text = the_text.replace('SETTING:',     '\n')
    the_text = the_text.replace('AIM:',         '\n')
    the_text = the_text.replace('MOTIVATION',   ' ')
    the_text = the_text.replace('SUMMARY',      ' ')
    the_text = the_text.replace('AIM',          ' ')
    the_text = the_text.replace('CONTACT',      ' ')
    the_text = the_text.replace('DATA SOURCES', ' ')
    the_text = the_text.replace('. .',          '.')
    the_text = the_text.replace('á',            'α')
    the_text = the_text.replace('&quot;',       '"')
    the_text = the_text.replace('&amp;',        '&')
    the_text = the_text.replace(' &lt; ',       '<')
    the_text = the_text.replace(' &gt; ',       '>')
    the_text = the_text.replace('&lt;',         '<')
    the_text = the_text.replace('&gt;',         '>')
    the_text = the_text.replace('>>',           '>')
    the_text = the_text.replace('<<',           '<')
    the_text = the_text.replace(' > ',          '>')
    the_text = the_text.replace(' < ',          '<')
    the_text = the_text.replace('.).',          '. ).')
    the_text = the_text.replace('AVAILABILITY AND IMPLEMENTATION',  ' ')
    the_text = the_text.replace('EXPERT OPINION',                   ' ')
    the_text = the_text.replace('Alzheimer\'s', 'Alzheimers')
    the_text = the_text.replace('Parkinson\'s', 'Parkinsons')
    the_text = the_text.replace('Hodgkin\'s',   'Hodgkins')
    the_text = the_text.replace('(?; ?)', '(? ; ?)')
    the_text = the_text.replace('(? ; ?)', '(? ; ? )')
    the_text = the_text.replace('(the wearing off" phenomenon)', '(the \'wearing off" phenomenon)')
    the_text = re.sub('\s+', ' ', the_text)
    the_text = the_text.strip()
    return the_text

def prepare_gold_dato(gold_dato):
    ret = {
        'body'      : gold_dato['body'],
        'documents' : gold_dato["documents"],
        'id'        : gold_dato["id"],
        'type'      : gold_dato["type"],
        'snippets'  : []

    }
    for snip in gold_dato['snippets']:
        did                     = snip['document'].split('/')[-1]
        fpath                   = '/home/dpappas/for_ryan/downloaded/{}.json'.format(did)
        dato                    = json.load(open(fpath))
        if(len(dato)==0):
            return None
        if('ArticleTitle' not in dato and 'AbstractText' not in dato):
            return None
        snip[u'text']           = fix_text_for_matching(snip[u'text'])
        # dato['ArticleTitle']    = '\n'.join(get_sents(dato['ArticleTitle']))
        dato['ArticleTitle']    = fix_text_for_matching(dato['ArticleTitle'])
        # dato['AbstractText']    = '\n'.join(get_sents(dato['AbstractText']))
        dato['AbstractText']    = fix_text_for_matching(dato['AbstractText'])
        for snip_sent in get_sents(snip[u'text']):
            tt = {
                "beginSection"  : snip["beginSection"],
                "endSection"    : snip["beginSection"],
                "text"          : snip_sent,
                "document"      : snip["document"]
            }
            if(snip["beginSection"] == 'title'):
                try:
                    tt['offsetInBeginSection']  = dato['ArticleTitle'].index(snip_sent)
                    tt['offsetInEndSection']    = dato['ArticleTitle'].index(snip_sent)+len(snip_sent)
                    ret['snippets'].append(tt)
                except:
                    print(snip["document"])
                    print(20 * '-')
                    print(snip[u'text'])
                    print(20 * '-')
                    print(snip_sent)
                    print(20 * '-')
                    print(dato['ArticleTitle'])
                    print(20 * '=')
            else:
                try:
                    tt['offsetInBeginSection']  = dato['AbstractText'].index(snip_sent)
                    tt['offsetInEndSection']    = dato['AbstractText'].index(snip_sent)+len(snip_sent)
                    ret['snippets'].append(tt)
                except:
                    print(snip[u'text'])
                    print(20 * '-')
                    print(snip["document"])
                    print(20 * '-')
                    print(snip_sent)
                    print(20 * '-')
                    print(dato['AbstractText'])
                    print(20 * '=')
    return ret

def eval_bioasq_snippets(prefix, data, docs):
    model.eval()
    all_bioasq_subm_data = {'questions':[]}
    all_bioasq_gold_data = {'questions':[]}
    for dato in tqdm(data['queries']):
        quest                       = dato['query_text']
        quest_tokens, quest_embeds  = get_embeds(tokenize(quest), wv)
        q_idfs                      = np.array([[idf_val(qw)] for qw in quest_tokens], 'float')
        doc_res                     = {}
        gold_snips                  = get_gold_snips(dato['query_id'])
        #
        pseudo_retrieved            = get_pseudo_retrieved(dato)
        extracted_snippets          = []
        for retr in pseudo_retrieved:
            good_res = handle_good(retr, quest)
            if(good_res is None):
                continue
            good_sents_embeds, good_sents_escores, good_doc_af, good_mesh_embeds, held_out_sents = good_res
            doc_emit_, gs_emits_    = model.emit_one(doc1_sents_embeds = good_sents_embeds, question_embeds = quest_embeds, q_idfs = q_idfs, sents_gaf = good_sents_escores, doc_gaf = good_doc_af, good_mesh_embeds = good_mesh_embeds)
            emitss                  = gs_emits_[:, 0].tolist()
            mmax    = max(emitss)
            indices = [
                item[0]
                for item in zip(range(len(emitss)), emitss)
                if(item[1] >= .6 or item[1] == mmax)
            ]
            for ind in indices:
                to_append = (
                        snip_is_relevant(held_out_sents[ind][1], gold_snips),
                        emitss[ind],
                        "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(retr['doc_id']),
                        held_out_sents[ind]
                    )
                extracted_snippets.append(to_append)
            #
            emition                 = doc_emit_.cpu().item()
            doc_res[retr['doc_id']] = float(emition)
        extracted_snippets          = sorted(extracted_snippets, key=lambda x: x[1], reverse=True)
        doc_res                     = sorted(doc_res.items(),    key=lambda x: x[1], reverse=True)
        doc_res                     = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
        #
        bioasq_subm_dato            = prep_bioasq_subm_dato(extracted_snippets, dato, doc_res)
        all_bioasq_subm_data['questions'].append(bioasq_subm_dato)
        #
        gold_dato   = bioasq6_data[dato['query_id']]
        gold_dato   = prepare_gold_dato(gold_dato)
        if(gold_dato is not None):
            all_bioasq_gold_data['questions'].append(gold_dato)
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data)
    return bioasq_snip_res

def get_one_map(prefix, data, docs):
    model.eval()
    ret_data                = {}
    ret_data['questions']   = []
    for dato in tqdm(data['queries']):
        quest                       = dato['query_text']
        quest_tokens, quest_embeds  = get_embeds(tokenize(quest), wv)
        q_idfs                      = np.array([[idf_val(qw)] for qw in quest_tokens], 'float')
        emitions                    = {'body': dato['query_text'], 'id': dato['query_id'], 'documents': []}
        bm25s                       = {t['doc_id']: t['norm_bm25_score'] for t in dato[u'retrieved_documents']}
        doc_res                     = {}
        for retr in dato['retrieved_documents']:
            #
            good_doc_text   = docs[retr['doc_id']]['title'] + docs[retr['doc_id']]['abstractText']
            good_doc_af     = GetScores(quest, good_doc_text, bm25s[retr['doc_id']])
            #
            good_sents      = get_sents(docs[retr['doc_id']]['title']) + get_sents(docs[retr['doc_id']]['abstractText'])
            good_sents_embeds, good_sents_escores = [], []
            held_out_sents  = []
            for good_text in good_sents:
                good_tokens, good_embeds = get_embeds(tokenize(good_text), wv)
                good_escores = GetScores(quest, good_text, bm25s[retr['doc_id']])[:-1]
                if (len(good_embeds) > 0):
                    good_sents_embeds.append(good_embeds)
                    good_sents_escores.append(good_escores)
                    held_out_sents.append(good_text)
            good_mesh               = get_the_mesh(docs[retr['doc_id']])
            gmt, good_mesh_embeds   = get_embeds(good_mesh, wv)
            #
            doc_emit_, gs_emits_    = model.emit_one(
                doc1_sents_embeds   = good_sents_embeds,
                question_embeds     = quest_embeds,
                q_idfs              = q_idfs,
                sents_gaf           = good_sents_escores,
                doc_gaf             = good_doc_af,
                good_mesh_embeds    = good_mesh_embeds
            )
            emition                 = doc_emit_.cpu().item()
            doc_res[retr['doc_id']] = float(emition)
        doc_res                     = sorted(doc_res.items(),    key=lambda x: x[1], reverse=True)
        doc_res                     = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
        emitions['documents']       = doc_res[:100]
        ret_data['questions'].append(emitions)
    if (prefix == 'dev'):
        with open(odir + 'elk_relevant_abs_posit_drmm_lists_dev.json', 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(dataloc+'bioasq.dev.json', odir + 'elk_relevant_abs_posit_drmm_lists_dev.json')
    else:
        with open(odir + 'elk_relevant_abs_posit_drmm_lists_test.json', 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(dataloc+'bioasq.test.json', odir + 'elk_relevant_abs_posit_drmm_lists_test.json')
    return res_map

def get_snippets_loss(good_sent_tags, gs_emits_, bs_emits_):
    wright = torch.cat([gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 1)])
    wrong  = [gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 0)]
    wrong  = torch.cat(wrong + [bs_emits_.squeeze(-1)])
    losses = [ model.my_hinge_loss(w.unsqueeze(0).expand_as(wrong), wrong) for w in wright]
    return sum(losses) / float(len(losses))

def get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_):
    bs_emits_       = bs_emits_.squeeze(-1)
    gs_emits_       = gs_emits_.squeeze(-1)
    good_sent_tags  = torch.FloatTensor(good_sent_tags)
    #
    sn_d1_l         = F.binary_cross_entropy(gs_emits_, good_sent_tags, size_average=False, reduce=True)
    sn_d2_l         = F.binary_cross_entropy(bs_emits_, torch.zeros_like(bs_emits_), size_average=False, reduce=True)
    return sn_d1_l, sn_d2_l

def train_one(epoch):
    model.train()
    batch_costs, batch_acc, epoch_costs, epoch_acc = [], [], [], []
    batch_counter = 0
    train_instances = train_data_step1()
    # train_instances = train_instances[:len(train_instances)/2]
    epoch_aver_cost, epoch_aver_acc = 0., 0.
    random.shuffle(train_instances)
    # for instance in train_data_step2(train_instances[:90*50]):
    for instance in train_data_step2(train_instances):
        cost_, doc1_emit_, doc2_emit_, gs_emits_, bs_emits_ = model(
            doc1_sents_embeds   = instance[0],
            doc2_sents_embeds   = instance[1],
            question_embeds     = instance[2],
            q_idfs              = instance[3],
            sents_gaf           = instance[4],
            sents_baf           = instance[5],
            doc_gaf             = instance[6],
            doc_baf             = instance[7],
            good_mesh_embeds    = instance[10],
            bad_mesh_embeds     = instance[11]
        )
        #
        good_sent_tags, bad_sent_tags       = instance[8], instance[9]
        sn_d1_l, sn_d2_l                    = get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_)
        snip_loss                           = sn_d1_l + sn_d2_l
        l                                   = 0.5
        cost_                               = ((1 - l) * snip_loss) + (l * cost_)
        #
        batch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_costs.append(cost_.cpu().item())
        batch_costs.append(cost_)
        if (len(batch_costs) == b_size):
            batch_counter += 1
            batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc)
            print('{} {} {} {} {}'.format(batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc))
            logger.info('{} {} {} {} {}'.format( batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc))
            batch_costs, batch_acc = [], []
    if (len(batch_costs) > 0):
        batch_counter += 1
        batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc)
        print('{} {} {} {} {}'.format(batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc))
        logger.info('{} {} {} {} {}'.format(batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc))
    print('Epoch:{} aver_epoch_cost: {} aver_epoch_acc: {}'.format(epoch, epoch_aver_cost, epoch_aver_acc))
    logger.info('Epoch:{} aver_epoch_cost: {} aver_epoch_acc: {}'.format(epoch, epoch_aver_cost, epoch_aver_acc))

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_sim_info(sent, snippets):
    sent            = bioclean(sent)
    similarities    = [similar(sent, s) for s in snippets]
    max_sim         = max(similarities)
    index_of_max    = similarities.index(max_sim)
    return max_sim, index_of_max, snippets[index_of_max]

def init_the_logger(hdlr):
    if not os.path.exists(odir):
        os.makedirs(odir)
    od          = odir.split('/')[-1] # 'sent_posit_drmm_MarginRankingLoss_0p001'
    logger      = logging.getLogger(od)
    if(hdlr is not None):
        logger.removeHandler(hdlr)
    hdlr        = logging.FileHandler(odir+'model.log')
    formatter   = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim, k_for_maxpool, k_sent_maxpool):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k                                      = k_for_maxpool         # k is for the average k pooling
        self.k2                                     = k_sent_maxpool        # k is for the average k pooling
        #
        self.embedding_dim                          = embedding_dim
        self.trigram_conv                           = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation                = torch.nn.LeakyReLU(negative_slope=0.1)
        self.q_weights_mlp                          = nn.Linear(self.embedding_dim+1, 1, bias=True)
        self.linear_per_q1                          = nn.Linear(6, 8, bias=True)
        self.my_relu1                               = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2                          = nn.Linear(8, 1, bias=True)
        self.margin_loss                            = nn.MarginRankingLoss(margin=1.0)
        self.out_layer                              = nn.Linear(4, 1, bias=True)
        # self.final_layer                            = nn.Linear(self.k2, 1, bias=True)
        self.final_layer                            = nn.Linear(5 + 10, 1, bias=True)
        #
        # num_layers * num_directions, batch, hidden_size
        self.mesh_h0                                = autograd.Variable(torch.randn(1, 1, 10))
        self.mesh_gru                               = nn.GRU(self.embedding_dim, 10)
        # self.init_xavier()
        # self.init_using_value(0.1)
        # MultiMarginLoss
        # MarginRankingLoss
        # my hinge loss
        # MultiLabelMarginLoss
        #
    def init_xavier(self):
        nn.init.xavier_uniform_(self.trigram_conv.weight)
        nn.init.xavier_uniform_(self.q_weights_mlp.weight)
        nn.init.xavier_uniform_(self.linear_per_q1.weight)
        nn.init.xavier_uniform_(self.linear_per_q2.weight)
        nn.init.xavier_uniform_(self.out_layer.weight)
    def init_using_value(self, value):
        self.trigram_conv.weight.data.fill_(value)
        self.q_weights_mlp.weight.data.fill_(value)
        self.linear_per_q1.weight.data.fill_(value)
        self.linear_per_q2.weight.data.fill_(value)
        self.out_layer.weight.data.fill_(value)
        self.trigram_conv.bias.data.fill_(value)
        self.q_weights_mlp.bias.data.fill_(value)
        self.linear_per_q1.bias.data.fill_(value)
        self.linear_per_q2.bias.data.fill_(value)
        self.out_layer.weight.data.fill_(value)
        self.final_layer.weight.data.fill_(value)
    def min_max_norm(self, x):
        minn        = torch.min(x)
        maxx        = torch.max(x)
        minmaxnorm  = (x-minn) / (maxx - minn)
        return minmaxnorm
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
    def do_for_one_doc(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights):
        res = []
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            conv_res            = self.apply_convolution(sent_embeds, self.trigram_conv, self.trigram_conv_activation)
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
            sent_out            = self.out_layer(sent_add_feats)
            res.append(sent_out)
        res = torch.stack(res)
        res = F.sigmoid(res)
        ret = self.get_max(res).unsqueeze(0)
        return ret, res
    def get_max_and_average_of_k_max(self, res, k):
        sorted_res              = torch.sort(res)[0]
        k_max_pooled            = sorted_res[-k:]
        average_k_max_pooled    = k_max_pooled.sum()/float(k)
        the_maximum             = k_max_pooled[-1]
        # print(the_maximum)
        # print(the_maximum.size())
        # print(average_k_max_pooled)
        # print(average_k_max_pooled.size())
        the_concatenation       = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation
    def get_max(self, res):
        return torch.max(res)
    def get_kmax(self, res):
        res     = torch.sort(res,0)[0]
        res     = res[-self.k2:].squeeze(-1)
        if(res.size()[0] < self.k2):
            res         = torch.cat([res, torch.zeros(self.k2 - res.size()[0])], -1)
        return res
    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res
    def get_maxmin_max(self, res):
        res = self.min_max_norm(res)
        res = torch.max(res)
        return res
    def apply_mesh_gru(self, mesh_embeds):
        # print(mesh_embeds.shape)
        mesh_embeds     = autograd.Variable(torch.FloatTensor(mesh_embeds), requires_grad=False)
        output, hn      = self.mesh_gru(mesh_embeds.unsqueeze(1), self.mesh_h0)
        return output[-1,0,:]
    def emit_one(self, doc1_sents_embeds, question_embeds, q_idfs, sents_gaf, doc_gaf, good_mesh_embeds):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds), requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False)
        q_conv_res_trigram  = self.apply_convolution(question_embeds, self.trigram_conv, self.trigram_conv_activation)
        q_weights           = torch.cat([q_conv_res_trigram, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        good_out, gs_emits  = self.do_for_one_doc(doc1_sents_embeds, sents_gaf, question_embeds, q_conv_res_trigram, q_weights)
        good_mesh_out       = self.apply_mesh_gru(good_mesh_embeds)
        good_out_pp         = torch.cat([good_out, doc_gaf, good_mesh_out], -1)
        final_good_output   = self.final_layer(good_out_pp)
        return final_good_output, gs_emits
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, question_embeds, q_idfs, sents_gaf, sents_baf, doc_gaf, doc_baf, good_mesh_embeds, bad_mesh_embeds):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        doc_baf             = autograd.Variable(torch.FloatTensor(doc_baf),             requires_grad=False)
        #
        q_conv_res_trigram  = self.apply_convolution(question_embeds, self.trigram_conv, self.trigram_conv_activation)
        q_weights           = torch.cat([q_conv_res_trigram, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits  = self.do_for_one_doc(doc1_sents_embeds, sents_gaf, question_embeds, q_conv_res_trigram, q_weights)
        bad_out,  bs_emits  = self.do_for_one_doc(doc2_sents_embeds, sents_baf, question_embeds, q_conv_res_trigram, q_weights)
        #
        good_mesh_out       = self.apply_mesh_gru(good_mesh_embeds)
        bad_mesh_out        = self.apply_mesh_gru(bad_mesh_embeds)
        #
        good_out_pp         = torch.cat([good_out, doc_gaf, good_mesh_out], -1)
        bad_out_pp          = torch.cat([bad_out,  doc_baf, bad_mesh_out],  -1)
        #
        final_good_output   = self.final_layer(good_out_pp)
        final_bad_output    = self.final_layer(bad_out_pp)
        #
        loss1               = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output, gs_emits, bs_emits

w2v_bin_path    = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
idf_pickle_path = '/home/dpappas/for_ryan/fordp/idf.pkl'
dataloc         = '/home/dpappas/for_ryan/'
eval_path       = '/home/dpappas/for_ryan/eval/run_eval.py'

# w2v_bin_path    = '/home/dpappas/for_ryan/pubmed2018_w2v_30D.bin'
# idf_pickle_path = '/home/dpappas/for_ryan/idf.pkl'
# dataloc         = '/home/DATA/Biomedical/document_ranking/bioasq_data/'
# eval_path       = '/home/DATA/Biomedical/document_ranking/eval/run_eval.py'

k_for_maxpool   = 5
k_sent_maxpool  = 2
embedding_dim   = 30 #200
lr              = 0.01
b_size          = 32
max_epoch       = 15

hdlr = None
for run in range(5):
    #
    my_seed = random.randint(1, 2000000)
    random.seed(my_seed)
    torch.manual_seed(my_seed)
    #
    odir            = '/home/dpappas/pdrmm_w2v_2L_eval_sents_run{}/'.format(run)
    #
    logger, hdlr    = init_the_logger(hdlr)
    print('random seed: {}'.format(my_seed))
    logger.info('random seed: {}'.format(my_seed))
    #
    (
        test_data, test_docs, dev_data, dev_docs,
        train_data, train_docs, idf, max_idf, wv,
        bioasq6_data
    ) = load_all_data(
        dataloc         = dataloc,
        w2v_bin_path    = w2v_bin_path,
        idf_pickle_path = idf_pickle_path
    )
    #
    print('Compiling model...')
    logger.info('Compiling model...')
    model       = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool, k_sent_maxpool=k_sent_maxpool)
    params      = model.parameters()
    print_params(model)
    optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #
    best_dev_map, test_map = None, None
    for epoch in range(max_epoch):
        train_one(epoch + 1)
        epoch_dev_map       = get_one_map('dev', dev_data, dev_docs)
        dev_bioasq_snip_res = eval_bioasq_snippets('dev', dev_data, dev_docs)
        pprint(dev_bioasq_snip_res)
        if(best_dev_map is None or epoch_dev_map>=best_dev_map):
            best_dev_map    = epoch_dev_map
            test_map        = get_one_map('test', test_data, test_docs)
            save_checkpoint(epoch, model, best_dev_map, optimizer, filename=odir+'best_checkpoint.pth.tar')
        test_bioasq_snip_res = eval_bioasq_snippets('test', test_data, test_docs)
        pprint(test_bioasq_snip_res)
        print('epoch:{} epoch_dev_map:{} best_dev_map:{} test_map:{}'.format(epoch + 1, epoch_dev_map, best_dev_map, test_map))
        logger.info('epoch:{} epoch_dev_map:{} best_dev_map:{} test_map:{}'.format(epoch + 1, epoch_dev_map, best_dev_map, test_map))

'''
Precision
Recall
F1
MAP
GMAP
MRR
'''

'''
Task B
---------

1. For running the measures for Task B, phase A the following command is invoked:

java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseA -e 5 golden_file.json system_response.json

2. For running the measures for Task B, phase B the following command is invoked:

java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 golden_file.json system_response.json



java -Xmx10G -cp \
/home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
/home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/BioASQ-task6bPhaseB-testset1 \
/home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/drmm-experimental_submit.json

'''

'''

java -Xmx10G -cp /home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseA -e 5 ./gold_bioasq.json emit_bioasq.json 

# stats on positions of relevant snippets
# add eval of snippets
add document score as well (original output of pdrmm)
treat mesh terms as sentences with pdrmm
add an rrn but keep pooling as well
# dump positional embeds for now
do everything with and without loss 2

'''


