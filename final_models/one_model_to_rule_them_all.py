#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys

reload(sys)
sys.setdefaultencoding("utf-8")

import  os
import  re
import  json
import  time
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

def get_map_res(fgold, femit):
    trec_eval_res   = subprocess.Popen(['python', eval_path, fgold, femit], stdout=subprocess.PIPE, shell=False)
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
              ' '.join(
                  [
                      ' '.join(mm) for mm in
                      get_the_mesh(doc_text[doc_id])
                  ]
              )
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
                good_snips.extend(sent_tokenize(sn['text']))
    return good_snips

def get_the_mesh(the_doc):
    good_meshes = []
    if('meshHeadingsList' in the_doc):
        for t in the_doc['meshHeadingsList']:
            t = t.split(':', 1)
            t = t[1].strip()
            t = t.lower()
            good_meshes.append(t)
    elif('MeshHeadings' in the_doc):
        for mesh_head_set in the_doc['MeshHeadings']:
            for item in mesh_head_set:
                good_meshes.append(item['text'].strip().lower())
    if('Chemicals' in the_doc):
        for t in the_doc['Chemicals']:
            t = t['NameOfSubstance'].strip().lower()
            good_meshes.append(t)
    good_mesh = sorted(good_meshes)
    good_mesh = ['mgmx'] + good_mesh
    # good_mesh = ' # '.join(good_mesh)
    # good_mesh = good_mesh.split()
    # good_mesh = [gm.split() for gm in good_mesh]
    good_mesh = [gm for gm in good_mesh]
    return good_mesh

def train_data_step2(train_instances):
    for quest, quest_id, gid, bid, bm25s_gid, bm25s_bid in train_instances:
        quest_tokens, quest_embeds              = get_embeds(tokenize(quest), wv)
        q_idfs                                  = np.array([[idf_val(qw)] for qw in quest_tokens], 'float')
        #
        good_meshes                             = get_the_mesh(train_docs[gid])
        good_doc_text                           = train_docs[gid]['title'] + train_docs[gid]['abstractText']
        good_doc_af                             = GetScores(quest, good_doc_text, bm25s_gid)
        good_sents_title                        = sent_tokenize(train_docs[gid]['title'])
        good_sents_abs                          = sent_tokenize(train_docs[gid]['abstractText'])
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
                tt          = ' '.join(bioclean(good_text))
                good_sent_tags.append(snip_is_relevant(tt, good_snips))
                # sims        = [similar(gs, tt) for gs in good_snips]
                # best_sim    = max(sims) if(len(sims)>0) else 0.
                # good_sent_tags.append(int(best_sim>0.9))
        # Handle good mesh terms
        good_mesh_embeds, good_mesh_escores = [], []
        for good_mesh in good_meshes:
            gm_tokens, gm_embeds = get_embeds(good_mesh, wv)
            if(len(gm_tokens)>0):
                good_mesh_embeds.append(gm_embeds)
                good_escores = GetScores(quest, good_mesh, bm25s_gid)[:-1]
                good_mesh_escores.append(good_escores)
        #
        bad_meshes                              = get_the_mesh(train_docs[bid])
        bad_doc_text                            = train_docs[bid]['title'] + train_docs[bid]['abstractText']
        bad_doc_af                              = GetScores(quest, bad_doc_text, bm25s_bid)
        bad_sents                               = sent_tokenize(train_docs[bid]['title']) + sent_tokenize(train_docs[bid]['abstractText'])
        #
        bad_sent_tags                           = len(bad_sents) * [0]
        bad_sents_embeds, bad_sents_escores     = [], []
        for bad_text in bad_sents:
            bad_tokens, bad_embeds              = get_embeds(tokenize(bad_text), wv)
            bad_escores                         = GetScores(quest, bad_text, bm25s_bid)[:-1]
            if(len(bad_embeds)>0):
                bad_sents_embeds.append(bad_embeds)
                bad_sents_escores.append(bad_escores)
        # Handle bad mesh terms
        bad_mesh_embeds, bad_mesh_escores = [], []
        for bad_mesh in bad_meshes:
            gm_tokens, gm_embeds = get_embeds(bad_mesh, wv)
            if(len(gm_tokens)>0):
                bad_mesh_embeds.append(gm_embeds)
                bad_escores = GetScores(quest, bad_mesh, bm25s_gid)[:-1]
                bad_mesh_escores.append(bad_escores)
        if(sum(good_sent_tags)>0):
            yield (
                good_sents_embeds,  bad_sents_embeds,   quest_embeds,       q_idfs,
                good_sents_escores, bad_sents_escores,  good_doc_af,        bad_doc_af,
                good_sent_tags,     bad_sent_tags,      good_mesh_embeds,   bad_mesh_embeds,
                good_mesh_escores,  bad_mesh_escores
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

def snip_is_relevant(one_sent, gold_snips):
    return any(
        [
            (one_sent.encode('ascii','ignore')  in gold_snip.encode('ascii','ignore'))
            or
            (gold_snip.encode('ascii','ignore') in one_sent.encode('ascii','ignore'))
            for gold_snip in gold_snips
        ]
    )
    # return max(
    #     [
    #         similar(one_sent, gold_snip)
    #         for gold_snip in gold_snips
    #     ]
    # )

def prep_data(quest, the_doc, the_bm25):
    good_doc_text   = the_doc['title'] + the_doc['abstractText']
    good_doc_af     = GetScores(quest, good_doc_text, the_bm25)
    good_sents      = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
    good_sents_embeds, good_sents_escores, held_out_sents = [], [], []
    for good_text in good_sents:
        good_tokens, good_embeds = get_embeds(tokenize(good_text), wv)
        good_escores = GetScores(quest, good_text, the_bm25)[:-1]
        if (len(good_embeds) > 0):
            good_sents_embeds.append(good_embeds)
            good_sents_escores.append(good_escores)
            held_out_sents.append(good_text)
    good_meshes         = get_the_mesh(the_doc)
    good_mesh_embeds, good_mesh_escores = [], []
    for good_mesh in good_meshes:
        gm_tokens, gm_embeds = get_embeds(good_mesh, wv)
        if (len(gm_tokens) > 0):
            good_mesh_embeds.append(gm_embeds)
            good_escores = GetScores(quest, good_mesh, the_bm25)[:-1]
            good_mesh_escores.append(good_escores)
    return good_sents_embeds, good_sents_escores, good_doc_af, good_mesh_embeds, held_out_sents, good_mesh_escores

def get_gold_snips(quest_id):
    gold_snips                  = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            gold_snips.extend(sent_tokenize(sn['text']))
    return list(set(gold_snips))

def prep_extracted_snippets(extracted_snippets, docs, qid, top10docs, quest_body):
    ret = {
        'body'      : quest_body,
        'documents' : top10docs,
        'id'        : qid,
        'snippets'  : [],
    }
    for esnip in extracted_snippets:
        pid         = esnip[2].split('/')[-1]
        the_text    = esnip[3]
        esnip_res = {
            # 'score'     : esnip[1],
            "document"  : "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pid),
            "text"      : the_text
        }
        try:
            ind_from    = docs[pid]['title'].index(the_text)
            ind_to      = ind_from + len(the_text)
            esnip_res["beginSection"]           = "title"
            esnip_res["endSection"]             = "title"
            esnip_res["offsetInBeginSection"]   = ind_from
            esnip_res["offsetInEndSection"]     = ind_to
        except:
            ind_from    = docs[pid]['abstractText'].index(the_text)
            ind_to      = ind_from + len(the_text)
            esnip_res["beginSection"]           = "abstract"
            esnip_res["endSection"]             = "abstract"
            esnip_res["offsetInBeginSection"]   = ind_from
            esnip_res["offsetInEndSection"]     = ind_to
        ret['snippets'].append(esnip_res)
    return ret

def get_bioasq_res(prefix, data_gold, data_emitted, data_for_revision):
    '''
    java -Xmx10G -cp /home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar
    evaluation.EvaluatorTask1b -phaseA -e 5
    /home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/BioASQ-task6bPhaseB-testset1
    ./drmm-experimental_submit.json
    '''
    jar_path = retrieval_jar_path
    #
    fgold   = '{}_data_for_revision.json'.format(prefix)
    fgold   = os.path.join(odir, fgold)
    fgold   = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_for_revision, indent=4, sort_keys=True))
        f.close()
    #
    for tt in data_gold['questions']:
        if ('exact_answer' in tt):
            del (tt['exact_answer'])
        if ('ideal_answer' in tt):
            del (tt['ideal_answer'])
        if ('type' in tt):
            del (tt['type'])
    fgold    = '{}_gold_bioasq.json'.format(prefix)
    fgold   = os.path.join(odir, fgold)
    fgold   = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_gold, indent=4, sort_keys=True))
        f.close()
    #
    femit    = '{}_emit_bioasq.json'.format(prefix)
    femit   = os.path.join(odir, femit)
    femit   = os.path.abspath(femit)
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
        if(':' in line):
            k       = line.split(':')[0].strip()
            v       = line.split(':')[1].strip()
            ret[k]  = float(v)
    return ret

def do_for_one_retrieved(quest, q_idfs, quest_embeds, bm25s, docs, retr, doc_res, gold_snips):
    doc = docs[retr['doc_id']]
    bm  = bm25s[retr['doc_id']]
    (
        good_sents_embeds, good_sents_escores, good_doc_af,
        good_meshes_embeds, held_out_sents, good_mesh_escores
    ) = prep_data(quest, doc, bm)
    doc_emit_, gs_emits_    = model.emit_one(
        doc1_sents_embeds   = good_sents_embeds,
        question_embeds     = quest_embeds,
        q_idfs              = q_idfs,
        sents_gaf           = good_sents_escores,
        doc_gaf             = good_doc_af,
        good_meshes_embeds  = good_meshes_embeds,
        mesh_gaf            = good_mesh_escores
    )
    emition                 = doc_emit_.cpu().item()
    emitss                  = gs_emits_.tolist()
    mmax                    = max(emitss)
    all_emits, extracted_from_one = [], []
    for ind in range(len(emitss)):
        t = (
            snip_is_relevant(held_out_sents[ind], gold_snips),
            emitss[ind],
            "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(retr['doc_id']),
            held_out_sents[ind]
        )
        all_emits.append(t)
        if(emitss[ind] == mmax):
            extracted_from_one.append(t)
    doc_res[retr['doc_id']] = float(emition)
    all_emits = sorted(all_emits, key=lambda x: x[1], reverse=True)
    return doc_res, extracted_from_one, all_emits

def similar(upstream_seq, downstream_seq):
    upstream_seq    = upstream_seq.encode('ascii','ignore')
    downstream_seq  = downstream_seq.encode('ascii','ignore')
    s               = SequenceMatcher(None, upstream_seq, downstream_seq)
    match           = s.find_longest_match(0, len(upstream_seq), 0, len(downstream_seq))
    upstream_start  = match[0]
    upstream_end    = match[0]+match[2]
    longest_match   = upstream_seq[upstream_start:upstream_end]
    to_match        = upstream_seq if(len(downstream_seq)>len(upstream_seq)) else downstream_seq
    r1              = SequenceMatcher(None, to_match, longest_match).ratio()
    return r1

def get_pseudo_retrieved(dato):
    some_ids = [item['document'].split('/')[-1].strip() for item in bioasq6_data[dato['query_id']]['snippets']]
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

def do_for_some_retrieved(docs, dato, retr_docs, data_for_revision, ret_data, all_bioasq_subm_data, all_bioasq_subm_data_known):
    quest                       = dato['query_text']
    quest_tokens, quest_embeds  = get_embeds(tokenize(quest), wv)
    q_idfs = np.array([[idf_val(qw)] for qw in quest_tokens], 'float')
    emitions = {
        'body': dato['query_text'],
        'id': dato['query_id'],
        'documents': []
    }
    bm25s = {t['doc_id']: t['norm_bm25_score'] for t in retr_docs}
    gold_snips = get_gold_snips(dato['query_id'])
    doc_res, extracted_snippets, extracted_snippets_known_rel_num = {}, [], []
    # for retr in get_pseudo_retrieved(dato):
    for retr in retr_docs:
        doc_res, extracted_from_one, all_emits = do_for_one_retrieved(
            quest, q_idfs, quest_embeds, bm25s, docs, retr,
            doc_res, gold_snips)
        extracted_snippets.extend(extracted_from_one)
        #
        total_relevant = sum([1 for em in all_emits if (em[0] == True)])
        if (total_relevant > 0):
            extracted_snippets_known_rel_num.extend(all_emits[:total_relevant])
        if (dato['query_id'] not in data_for_revision):
            data_for_revision[dato['query_id']] = {
                'query_text': dato['query_text'],
                'snippets': {retr['doc_id']: all_emits}
            }
        else:
            data_for_revision[dato['query_id']]['snippets'][retr['doc_id']] = all_emits
    doc_res = sorted(doc_res.items(), key=lambda x: x[1], reverse=True)
    doc_res = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
    emitions['documents'] = doc_res[:100]
    ret_data['questions'].append(emitions)
    #
    extracted_snippets = [tt for tt in extracted_snippets if (tt[2] in doc_res[:10])]
    extracted_snippets = sorted(extracted_snippets, key=lambda x: x[1], reverse=True)
    snips_res = prep_extracted_snippets(extracted_snippets, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    all_bioasq_subm_data['questions'].append(snips_res)
    #
    extracted_snippets_known_rel_num = [tt for tt in extracted_snippets_known_rel_num if (tt[2] in doc_res[:10])]
    extracted_snippets_known_rel_num = sorted(extracted_snippets_known_rel_num, key=lambda x: x[1], reverse=True)
    snips_res_known_rel_num = prep_extracted_snippets(extracted_snippets_known_rel_num, docs, dato['query_id'],
                                                      doc_res[:10], dato['query_text'])
    all_bioasq_subm_data_known['questions'].append(snips_res_known_rel_num)
    return data_for_revision, ret_data, all_bioasq_subm_data, all_bioasq_subm_data_known

def get_one_map(prefix, data, docs):
    model.eval()
    #
    ret_data                    = {'questions': []}
    all_bioasq_subm_data        = {"questions": []}
    all_bioasq_subm_data_known  = {"questions": []}
    all_bioasq_gold_data        = {'questions': []}
    data_for_revision           = {}
    #
    # ret_data_2                    = {'questions': []}
    # all_bioasq_subm_data_2        = {"questions": []}
    # all_bioasq_subm_data_known_2  = {"questions": []}
    # all_bioasq_gold_data_2        = {'questions': []}
    # data_for_revision_2           = {}
    for dato in tqdm(data['queries']):
        all_bioasq_gold_data['questions'].append(bioasq6_data[dato['query_id']])
        #
        data_for_revision, ret_data, all_bioasq_subm_data, all_bioasq_subm_data_known = do_for_some_retrieved(
            docs, dato, dato['retrieved_documents'],
            data_for_revision, ret_data, all_bioasq_subm_data, all_bioasq_subm_data_known
        )
        # for retr in get_pseudo_retrieved(dato):
        # data_for_revision_2, ret_data_2, all_bioasq_subm_data_2, all_bioasq_subm_data_known_2 = do_for_some_retrieved(
        #     docs, dato, get_pseudo_retrieved(dato),
        #     data_for_revision_2, ret_data_2, all_bioasq_subm_data_2, all_bioasq_subm_data_known_2
        # )
    #
    # bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data_2, all_bioasq_subm_data_known_2, data_for_revision_2)
    # pprint(bioasq_snip_res)
    # logger.info('{} gold docs known MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    # logger.info('{} gold docs known F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    # logger.info('{} gold docs known MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    # logger.info('{} gold docs known GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data_2, all_bioasq_subm_data_2, data_for_revision_2)
    pprint(bioasq_snip_res)
    logger.info('{} gold docs MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    logger.info('{} gold docs F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    logger.info('{} gold docs MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    logger.info('{} gold docs GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data_known, data_for_revision)
    pprint(bioasq_snip_res)
    logger.info('{} known MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    logger.info('{} known F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    logger.info('{} known MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    logger.info('{} known GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data, data_for_revision)
    pprint(bioasq_snip_res)
    logger.info('{} MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    logger.info('{} F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    logger.info('{} MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    logger.info('{} GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #
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

def train_one(epoch, two_losses=True):
    model.train()
    batch_costs, batch_acc, epoch_costs, epoch_acc = [], [], [], []
    batch_counter = 0
    train_instances = train_data_step1()
    # train_instances = train_instances[:len(train_instances)/2]
    epoch_aver_cost, epoch_aver_acc = 0., 0.
    random.shuffle(train_instances)
    # for instance in train_data_step2(train_instances[:90*50]):
    start_time      = time.time()
    for (
        good_sents_embeds,  bad_sents_embeds,   quest_embeds,       q_idfs,
        good_sents_escores, bad_sents_escores,  good_doc_af,        bad_doc_af,
        good_sent_tags,     bad_sent_tags,      good_mesh_embeds,   bad_mesh_embeds,
        good_mesh_escores,  bad_mesh_escores
    ) in train_data_step2(train_instances):
        cost_, doc1_emit_, doc2_emit_, gs_emits_, bs_emits_ = model(
            doc1_sents_embeds   = good_sents_embeds,
            doc2_sents_embeds   = bad_sents_embeds,
            question_embeds     = quest_embeds,
            q_idfs              = q_idfs,
            sents_gaf           = good_sents_escores,
            sents_baf           = bad_sents_escores,
            doc_gaf             = good_doc_af,
            doc_baf             = bad_doc_af,
            good_meshes_embeds  = good_mesh_embeds,
            bad_meshes_embeds   = bad_mesh_embeds,
            mesh_gaf            = good_mesh_escores,
            mesh_baf            = bad_mesh_escores
        )
        #
        good_sent_tags, bad_sent_tags       = good_sent_tags, bad_sent_tags
        if(two_losses):
            sn_d1_l, sn_d2_l                = get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_)
            snip_loss                       = sn_d1_l + sn_d2_l
            l                               = 0.5
            cost_                           = ((1 - l) * snip_loss) + (l * cost_)
        #
        batch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_costs.append(cost_.cpu().item())
        batch_costs.append(cost_)
        if (len(batch_costs) == b_size):
            batch_counter += 1
            batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc)
            elapsed_time    = time.time() - start_time
            start_time      = time.time()
            print('{} {} {} {} {} {}'.format(batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc, elapsed_time))
            logger.info('{} {} {} {} {} {}'.format( batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc, elapsed_time))
            batch_costs, batch_acc = [], []
    if (len(batch_costs) > 0):
        batch_counter += 1
        batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('{} {} {} {} {} {}'.format(batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc, elapsed_time))
        logger.info('{} {} {} {} {} {}'.format(batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc, elapsed_time))
    print('Epoch:{} aver_epoch_cost: {} aver_epoch_acc: {}'.format(epoch, epoch_aver_cost, epoch_aver_acc))
    logger.info('Epoch:{} aver_epoch_cost: {} aver_epoch_acc: {}'.format(epoch, epoch_aver_cost, epoch_aver_acc))

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
    def __init__(self, embedding_dim= 30, k_for_maxpool= 5, context_method = 'CNN', sentence_out_method = 'MLP', mesh_style = 'SENT'):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k                                      = k_for_maxpool
        #
        self.embedding_dim                          = embedding_dim
        self.mesh_style                             = mesh_style
        self.context_method                         = context_method
        self.sentence_out_method                    = sentence_out_method
        # to create q weights
        self.init_context_module()
        self.init_question_weight_module()
        self.init_mlps_for_pooled_attention()
        self.init_sent_output_layer()
        self.init_doc_out_layer()
        # doc loss func
        self.margin_loss                            = nn.MarginRankingLoss(margin=1.0)
    def init_mesh_module(self):
        self.mesh_h0    = autograd.Variable(torch.randn(1, 1, self.embedding_dim))
        self.mesh_gru   = nn.GRU(self.embedding_dim, self.embedding_dim)
    def init_context_module(self):
        if(self.context_method == 'CNN'):
            self.trigram_conv_1             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
            self.trigram_conv_activation_1  = torch.nn.LeakyReLU(negative_slope=0.1)
            self.trigram_conv_2             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
            self.trigram_conv_activation_2  = torch.nn.LeakyReLU(negative_slope=0.1)
        else:
            self.context_h0     = autograd.Variable(torch.randn(2, 1, self.embedding_dim))
            self.context_gru    = nn.GRU(
                input_size      = self.embedding_dim,
                hidden_size     = self.embedding_dim,
                bidirectional   = True
            )
            self.context_gru_activation = torch.nn.LeakyReLU(negative_slope=0.1)
    def init_question_weight_module(self):
        self.q_weights_mlp      = nn.Linear(self.embedding_dim+1, 1, bias=True)
    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1      = nn.Linear(3 * 3, 8, bias=True)
        self.my_relu1           = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2      = nn.Linear(8, 1, bias=True)
    def init_sent_output_layer(self):
        if(self.sentence_out_method == 'MLP'):
            self.sent_out_layer = nn.Linear(4, 1, bias=False)
        else:
            self.sent_res_h0    = autograd.Variable(torch.randn(2, 1, 5))
            self.sent_res_bigru = nn.GRU(input_size=4, hidden_size=5, bidirectional=True, batch_first=False)
            self.sent_res_mlp   = nn.Linear(10, 1, bias=False)
    def init_doc_out_layer(self):
        if(self.mesh_style=='BIGRU'):
            self.init_mesh_module()
            self.final_layer = nn.Linear(5 + 30, 1, bias=True)
        elif(self.mesh_style=='SENT'):
            self.final_layer = nn.Linear(1 + 4 + 1, 1, bias=True)
        else:
            self.final_layer = nn.Linear(5, 1, bias=True)
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
    def do_for_one_doc_cnn(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights):
        res = []
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            conv_res            = self.apply_context_convolution(sent_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
            conv_res            = self.apply_context_convolution(conv_res,      self.trigram_conv_2, self.trigram_conv_activation_2)
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
            res = self.sent_out_layer(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        ret = self.get_max(res).unsqueeze(0)
        res = torch.sigmoid(res)
        return ret, res
    def do_for_one_doc_bigru(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights):
        res = []
        hn  = self.context_h0
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
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
            res = self.sent_out_layer(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        ret = self.get_max(res).unsqueeze(0)
        res = torch.sigmoid(res)
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
        mesh_embeds     = autograd.Variable(torch.FloatTensor(mesh_embeds), requires_grad=False)
        output, hn      = self.mesh_gru(mesh_embeds.unsqueeze(1), self.mesh_h0)
        return output[-1,0,:]
    def get_mesh_rep(self, meshes_embeds, q_context):
        meshes_embeds   = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds   = torch.stack(meshes_embeds)
        sim_matrix      = self.my_cosine_sim(meshes_embeds, q_context).squeeze(0)
        max_sim         = torch.sort(sim_matrix, -1)[0][:, -1]
        output          = torch.mm(max_sim.unsqueeze(0), meshes_embeds)[0]
        return output
    def emit_one(self, doc1_sents_embeds, question_embeds, q_idfs, sents_gaf, doc_gaf, good_meshes_embeds, mesh_gaf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        #
        if(self.context_method=='CNN'):
            q_context       = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
            q_context       = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        else:
            q_context, _    = self.apply_context_gru(question_embeds, self.context_h0)
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        if(self.context_method=='CNN'):
            good_out, gs_emits  = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights)
        else:
            good_out, gs_emits = self.do_for_one_doc_bigru(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights)
        #
        if(self.mesh_style=='BIGRU'):
            good_meshes_out = self.get_mesh_rep(good_meshes_embeds, q_context)
            good_out_pp = torch.cat([good_out, doc_gaf, good_meshes_out], -1)
        elif (self.mesh_style == 'SENT'):
            if (self.context_method == 'CNN'):
                good_mesh_out, gs_mesh_emits = self.do_for_one_doc_cnn(good_meshes_embeds, mesh_gaf, question_embeds, q_context, q_weights)
            else:
                good_mesh_out, gs_mesh_emits = self.do_for_one_doc_bigru(good_meshes_embeds, mesh_gaf, question_embeds, q_context, q_weights)
            good_out_pp = torch.cat([good_out, doc_gaf, good_mesh_out], -1)
        else:
            good_out_pp = torch.cat([good_out, doc_gaf], -1)
        #
        final_good_output   = self.final_layer(good_out_pp)
        return final_good_output, gs_emits
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, question_embeds, q_idfs, sents_gaf, sents_baf, doc_gaf, doc_baf, good_meshes_embeds, bad_meshes_embeds, mesh_gaf, mesh_baf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        doc_baf             = autograd.Variable(torch.FloatTensor(doc_baf),             requires_grad=False)
        #
        if(self.context_method=='CNN'):
            q_context       = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
            q_context       = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        else:
            q_context, _    = self.apply_context_gru(question_embeds, self.context_h0)
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        if(self.context_method=='CNN'):
            good_out, gs_emits  = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights)
            bad_out, bs_emits   = self.do_for_one_doc_cnn(doc2_sents_embeds, sents_baf, question_embeds, q_context, q_weights)
        else:
            good_out, gs_emits  = self.do_for_one_doc_bigru(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights)
            bad_out, bs_emits   = self.do_for_one_doc_bigru(doc2_sents_embeds, sents_baf, question_embeds, q_context, q_weights)
        #
        if(self.mesh_style=='BIGRU'):
            good_meshes_out     = self.get_mesh_rep(good_meshes_embeds, q_context)
            bad_meshes_out      = self.get_mesh_rep(bad_meshes_embeds, q_context)
            good_out_pp         = torch.cat([good_out, doc_gaf, good_meshes_out], -1)
            bad_out_pp          = torch.cat([bad_out, doc_baf, bad_meshes_out], -1)
        elif(self.mesh_style=='SENT'):
            if(self.context_method=='CNN'):
                good_mesh_out, gs_mesh_emits = self.do_for_one_doc_cnn(
                    good_meshes_embeds, mesh_gaf, question_embeds, q_context, q_weights
                )
                bad_mesh_out, bs_mesh_emits = self.do_for_one_doc_cnn(
                    bad_meshes_embeds, mesh_baf, question_embeds, q_context, q_weights
                )
            else:
                good_mesh_out, gs_mesh_emits = self.do_for_one_doc_bigru(
                    good_meshes_embeds, mesh_gaf, question_embeds, q_context, q_weights
                )
                bad_mesh_out, bs_mesh_emits  = self.do_for_one_doc_bigru(
                    bad_meshes_embeds, mesh_baf, question_embeds, q_context, q_weights
                )
            good_out_pp = torch.cat([good_out, doc_gaf, good_mesh_out], -1)
            bad_out_pp  = torch.cat([bad_out, doc_baf, bad_mesh_out], -1)
        else:
            good_out_pp         = torch.cat([good_out, doc_gaf], -1)
            bad_out_pp          = torch.cat([bad_out, doc_baf], -1)
        #
        final_good_output   = self.final_layer(good_out_pp)
        final_bad_output    = self.final_layer(bad_out_pp)
        #
        loss1               = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output, gs_emits, bs_emits

w2v_bin_path        = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/for_ryan/fordp/idf.pkl'
dataloc             = '/home/dpappas/for_ryan/'
eval_path           = '/home/dpappas/for_ryan/eval/run_eval.py'
retrieval_jar_path  = '/home/dpappas/NetBeansProjects/my_bioasq_eval_2/dist/my_bioasq_eval_2.jar'

# w2v_bin_path        = '/home/dpappas/for_ryan/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/dpappas/for_ryan/idf.pkl'
# dataloc             = '/home/DATA/Biomedical/document_ranking/bioasq_data/'
# eval_path           = '/home/DATA/Biomedical/document_ranking/eval/run_eval.py'
# retrieval_jar_path  = '/home/dpappas/bioasq_eval/dist/my_bioasq_eval_2.jar'

# w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
# dataloc             = '/home/dpappas/bioasq_all/bioasq_data/'
# eval_path           = '/home/dpappas/bioasq_all/eval/run_eval.py'
# retrieval_jar_path  = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'

# w2v_bin_path        = '/home/cave/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/cave/dpappas/bioasq_all/idf.pkl'
# dataloc             = '/home/cave/dpappas/bioasq_all/bioasq_data/'
# eval_path           = '/home/cave/dpappas/bioasq_all/eval/run_eval.py'
# retrieval_jar_path  = '/home/cave/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'

k_for_maxpool   = 5
k_sent_maxpool  = 2
embedding_dim   = 30 #200
lr              = 0.01
b_size          = 32
max_epoch       = 10


models = [
# ['Model_01', ]
# ['Model_02', ],
# ['Model_03', ],
# ['Model_04', ],
# ['Model_05', ],
# ['Model_06', ],
# ['Model_07', ],
# ['Model_08', ],
# ['Model_09', 'CNN',     'MLP',   None,      False],
# ['Model_10', 'CNN',     'MLP',   'BIGRU',   False],
# ['Model_11', 'CNN',     'BIGRU', None,      False],
# ['Model_12', 'CNN',     'BIGRU', 'BIGRU',   False],
# ['Model_13', 'BIGRU',   'MLP',   None,      False],
# ['Model_14', 'BIGRU',   'MLP',   'BIGRU',   False],
# ['Model_15', 'BIGRU',   'BIGRU', None,      False],
# ['Model_16', 'BIGRU',   'BIGRU', 'BIGRU',   False],
# ['Model_17', 'CNN',     'MLP',   None,      True],
# ['Model_18', 'CNN',     'MLP',   'BIGRU',   True],
# ['Model_19', 'CNN',     'BIGRU', None,      True],
# ['Model_20', 'CNN',     'BIGRU', 'BIGRU',   True],
# ['Model_21', 'BIGRU',   'MLP',   None,      True],
# ['Model_22', 'BIGRU',   'MLP',   'BIGRU',   True],
# ['Model_23', 'BIGRU',   'BIGRU', None,      True],
# ['Model_24', 'BIGRU',   'BIGRU', 'BIGRU',   True],
# #
# ['Model_25', 'CNN',     'MLP',   'SENT',    False],
# ['Model_26', 'CNN',     'BIGRU', 'SENT',    False],
# ['Model_27', 'BIGRU',   'MLP',   'SENT',    False],
# ['Model_28', 'BIGRU',   'BIGRU', 'SENT',    False],
# ['Model_29', 'CNN',     'MLP',   'SENT',    True],
# ['Model_30', 'CNN',     'BIGRU', 'SENT',    True],
# ['Model_31', 'BIGRU',   'MLP',   'SENT',    True],
# ['Model_32', 'BIGRU',   'BIGRU', 'SENT',    True],
#
['Model_33', 'CNN',     'MLP',      None,      False],
['Model_34', 'CNN',     'MLP',      'SENT',    False],
['Model_35', 'CNN',     'BIGRU',    None,      False],
['Model_36', 'CNN',     'BIGRU',    'SENT',    False],
['Model_37', 'BIGRU',   'MLP',      None,      False],
['Model_38', 'BIGRU',   'MLP',      'SENT',    False],
['Model_39', 'BIGRU',   'BIGRU',    None,      False],
['Model_40', 'BIGRU',   'BIGRU',    'SENT',    False],
['Model_41', 'CNN',     'MLP',      None,      False],
['Model_42', 'CNN',     'MLP',      'SENT',    False],
['Model_43', 'CNN',     'BIGRU',    None,      False],
['Model_44', 'CNN',     'BIGRU',    'SENT',    False],
['Model_45', 'BIGRU',   'MLP',      None,      False],
['Model_46', 'BIGRU',   'MLP',      'SENT',    False],
['Model_47', 'BIGRU',   'BIGRU',    None,      False],
['Model_48', 'BIGRU',   'BIGRU',    'SENT',    False],
['Model_49', 'CNN',     'MLP',      None,      True],
['Model_50', 'CNN',     'MLP',      'SENT',    True],
['Model_51', 'CNN',     'BIGRU',    None,      True],
['Model_52', 'CNN',     'BIGRU',    'SENT',    True],
['Model_53', 'BIGRU',   'MLP',      None,      True],
['Model_54', 'BIGRU',   'MLP',      'SENT',    True],
['Model_55', 'BIGRU',   'BIGRU',    None,      True],
['Model_56', 'BIGRU',   'BIGRU',    'SENT',    True],


]
models = dict(
    [
        (item[0], item[1:])
        for item in models
    ]
)

which_model = 'Model_52'

hdlr = None
for run in range(5):
    #
    my_seed = random.randint(1, 2000000)
    random.seed(my_seed)
    torch.manual_seed(my_seed)
    #
    odir            = '/home/dpappas/{}_run_{}/'.format(which_model, run)
    #
    logger, hdlr    = init_the_logger(hdlr)
    print('random seed: {}'.format(my_seed))
    logger.info('random seed: {}'.format(my_seed))
    #
    (
        test_data, test_docs, dev_data, dev_docs, train_data,
        train_docs, idf, max_idf, wv, bioasq6_data
    ) = load_all_data(dataloc=dataloc, w2v_bin_path=w2v_bin_path, idf_pickle_path=idf_pickle_path)
    #
    print('Compiling model...')
    logger.info('Compiling model...')
    model       = Sent_Posit_Drmm_Modeler(
        embedding_dim       = embedding_dim,
        k_for_maxpool       = k_for_maxpool,
        context_method      = models[which_model][0],
        sentence_out_method = models[which_model][1],
        mesh_style          = models[which_model][2]
    )
    params      = model.parameters()
    print_params(model)
    optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #
    best_dev_map, test_map = None, None
    for epoch in range(max_epoch):
        # train_one(epoch + 1, two_losses=models[which_model][3])
        epoch_dev_map       = get_one_map('dev', dev_data, dev_docs)
        if(best_dev_map is None or epoch_dev_map>=best_dev_map):
            best_dev_map    = epoch_dev_map
            test_map        = get_one_map('test', test_data, test_docs)
            save_checkpoint(epoch, model, best_dev_map, optimizer, filename=odir+'best_checkpoint.pth.tar')
        print('epoch:{} epoch_dev_map:{} best_dev_map:{} test_map:{}'.format(epoch + 1, epoch_dev_map, best_dev_map, test_map))
        logger.info('epoch:{} epoch_dev_map:{} best_dev_map:{} test_map:{}'.format(epoch + 1, epoch_dev_map, best_dev_map, test_map))


'''
Petros
give the htmls to peter for annotation
keep the original text and keep what metamap thought

Sotiris


Me
remove the bias on snippet extraction before sigmoid    -- DONE
add average similarity score as 3rd feature             -- DONE
add oracle scores for snippets                          -- pending
compare to polivios scores                              -- pending

'''

# grep "test_map:" -B4  Model_14_run_4/model.log


'''
mail
ergasthria A48 apo epomenh
i wra askiseis mia kwdika
15 lepta dialleima. Sto 16 ksekinaw.
Posoi einai apo persi
posoi den kseroun python
posoi den kseroun java
posoi einai apo allo tmima
posoi to phrane giati einai efkolo
posoi to phrane giati theloun na kinigisoun ton tomea
ergasies 2 20% kathe mia (2-3 atoma)
oxi antigrafes
an argisw pote 10 lepta tis defteres mh shkwthoun na fygoun.
apories genikes
apories gia th prohgoumenh evdomada
ta ektos den paizoun (tupou den mporw na erthw Deftera kai Triti mporeite na mou ta peite alli mera/skype/til klp.)
'''


