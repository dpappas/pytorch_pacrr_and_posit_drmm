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

def load_all_data(w2v_bin_path, idf_pickle_path):
    print('loading pickle data')
    #
    # golden_data = pickle.load(open(golden,      'rb'))
    test_data   = pickle.load(open(retrieved,   'rb'))
    test_docs   = pickle.load(open(docs,        'rb'))
    test_data   = RemoveBadYears(test_data, test_docs, False)
    #
    words = {}
    GetWords(test_data, test_docs, words)
    print('loading idfs')
    idf, max_idf = load_idfs(idf_pickle_path, words)
    print('loading w2v')
    wv = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    wv = dict([(word, wv[word]) for word in wv.vocab.keys() if (word in words)])
    #
    return test_data, test_docs, idf, max_idf, wv

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
    good_mesh = [gm.split() for gm in good_mesh]
    return good_mesh

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
    good_mesh_embeds    = [get_embeds(good_mesh, wv)    for good_mesh           in good_meshes]
    good_mesh_embeds    = [good_mesh[1] for good_mesh   in  good_mesh_embeds    if(len(good_mesh[0])>0)]
    # gmt, good_mesh_embeds   = get_embeds(good_mesh, wv)
    return good_sents_embeds, good_sents_escores, good_doc_af, good_mesh_embeds, held_out_sents

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
    jar_path = '/home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar'
    #
    fgold    = './{}_data_for_revision.json'.format(prefix)
    fgold    = os.path.abspath(fgold)
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

def do_for_one_retrieved(quest, q_idfs, quest_embeds, bm25s, docs, retr, doc_res, gold_snips):
    (
        good_sents_embeds, good_sents_escores, good_doc_af,
        good_meshes_embeds, held_out_sents
    ) = prep_data(quest, docs[retr['doc_id']], bm25s[retr['doc_id']])
    doc_emit_, gs_emits_    = model.emit_one(
        doc1_sents_embeds   = good_sents_embeds,
        question_embeds     = quest_embeds,
        q_idfs              = q_idfs,
        sents_gaf           = good_sents_escores,
        doc_gaf             = good_doc_af,
        good_meshes_embeds  = good_meshes_embeds
    )
    emition                 = doc_emit_.cpu().item()
    emitss                  = gs_emits_[:, 0].tolist()
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

def get_one_map(prefix, data, docs):
    model.eval()
    ret_data                = {'questions': []}
    all_bioasq_subm_data    = {"questions": []}
    all_bioasq_gold_data    = {'questions': []}
    data_for_revision       = {}
    for dato in tqdm(data['queries']):
        all_bioasq_gold_data['questions'].append(bioasq6_data[dato['query_id']])
        quest                       = dato['query_text']
        quest_tokens, quest_embeds  = get_embeds(tokenize(quest), wv)
        q_idfs                      = np.array([[idf_val(qw)] for qw in quest_tokens], 'float')
        emitions                    = {
            'body'      : dato['query_text'],
            'id'        : dato['query_id'],
            'documents' : []
        }
        bm25s                       = { t['doc_id'] : t['norm_bm25_score'] for t in dato[u'retrieved_documents']}
        gold_snips                  = get_gold_snips(dato['query_id'])
        doc_res, extracted_snippets = {}, []
        # for retr in get_pseudo_retrieved(dato):
        for retr in dato['retrieved_documents']:
            doc_res, extracted_from_one, all_emits  = do_for_one_retrieved(quest, q_idfs, quest_embeds, bm25s, docs, retr, doc_res, gold_snips)
            extracted_snippets.extend(extracted_from_one)
            #
            if (dato['query_id'] not in data_for_revision):
                data_for_revision[dato['query_id']] = {
                    'query_text': dato['query_text'],
                    'snippets'  : {retr['doc_id']: all_emits}
                }
            else:
                data_for_revision[dato['query_id']]['snippets'][retr['doc_id']] = all_emits
        doc_res                     = sorted(doc_res.items(),    key=lambda x: x[1], reverse=True)
        doc_res                     = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
        emitions['documents']       = doc_res[:100]
        ret_data['questions'].append(emitions)
        #
        extracted_snippets                              = [tt for tt in extracted_snippets if(tt[2] in doc_res[:10])]
        extracted_snippets                              = sorted(extracted_snippets, key=lambda x: x[1], reverse=True)
        snips_res                                       = prep_extracted_snippets(extracted_snippets, docs, dato['query_id'], doc_res[:10], dato['query_text'])
        all_bioasq_subm_data['questions'].append(snips_res)
    #
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data, data_for_revision)
    pprint(bioasq_snip_res)
    #
    if (prefix == 'dev'):
        res_map = get_map_res(dataloc+'bioasq.dev.json', odir + 'elk_relevant_abs_posit_drmm_lists_dev.json')
    else:
        res_map = get_map_res(dataloc+'bioasq.test.json', odir + 'elk_relevant_abs_posit_drmm_lists_test.json')
    return res_map

def load_model_from_checkpoint(resume_from):
    global start_epoch, optimizer
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim, k_for_maxpool, k_sent_maxpool):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k                                      = k_for_maxpool         # k is for the average k pooling
        self.k2                                     = k_sent_maxpool        # k is for the average k pooling
        #
        self.embedding_dim                          = embedding_dim
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
        self.context_h0                             = autograd.Variable(torch.randn(2, 1, self.embedding_dim))
        self.context_gru                            = nn.GRU(
            input_size      = self.embedding_dim,
            hidden_size     = self.embedding_dim,
            bidirectional   = True
        )
        self.context_gru_activation                 = torch.nn.LeakyReLU(negative_slope=0.1)
        # num_layers * num_directions, batch, hidden_size
        self.mesh_h0_first                          = autograd.Variable(torch.randn(1, 1, 10))
        self.mesh_gru_first                         = nn.GRU(self.embedding_dim, 10)
        self.mesh_h0_second                         = autograd.Variable(torch.randn(1, 1, 10))
        self.mesh_gru_second                        = nn.GRU(10, 10)
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
    def apply_context_gru(self, the_input, h0):
        output, hn      = self.context_gru(the_input.unsqueeze(1), h0)
        output          = self.context_gru_activation(output)
        out_forward     = output[:, 0, :self.embedding_dim]
        out_backward    = output[:, 0, self.embedding_dim:]
        output          = out_forward + out_backward
        res             = output + the_input
        return res, hn
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
        mesh_embeds     = autograd.Variable(torch.FloatTensor(mesh_embeds), requires_grad=False)
        output, hn      = self.mesh_gru_first(mesh_embeds.unsqueeze(1), self.mesh_h0_first)
        return output[-1,0,:]
    def apply_stacked_mesh_gru(self, meshes_embeds):
        meshes_embeds   = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds   = torch.stack(meshes_embeds)
        output, hn      = self.mesh_gru_second(meshes_embeds.unsqueeze(1), self.mesh_h0_second)
        return output[-1, 0, :]
    def emit_one(self, doc1_sents_embeds, question_embeds, q_idfs, sents_gaf, doc_gaf, good_meshes_embeds):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds), requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False)
        q_gru_res, _        = self.apply_context_gru(question_embeds, self.context_h0)
        q_weights           = torch.cat([q_gru_res, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        good_out, gs_emits  = self.do_for_one_doc(doc1_sents_embeds, sents_gaf, question_embeds, q_gru_res, q_weights)
        good_meshes_out     = self.apply_stacked_mesh_gru(good_meshes_embeds)
        good_out_pp         = torch.cat([good_out, doc_gaf, good_meshes_out], -1)
        final_good_output   = self.final_layer(good_out_pp)
        return final_good_output, gs_emits
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, question_embeds, q_idfs, sents_gaf, sents_baf, doc_gaf, doc_baf, good_meshes_embeds, bad_meshes_embeds):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        doc_baf             = autograd.Variable(torch.FloatTensor(doc_baf),             requires_grad=False)
        #
        q_gru_res, _        = self.apply_context_gru(question_embeds, self.context_h0)
        q_weights           = torch.cat([q_gru_res, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits  = self.do_for_one_doc(doc1_sents_embeds, sents_gaf, question_embeds, q_gru_res, q_weights)
        bad_out,  bs_emits  = self.do_for_one_doc(doc2_sents_embeds, sents_baf, question_embeds, q_gru_res, q_weights)
        #
        good_meshes_out     = self.apply_stacked_mesh_gru(good_meshes_embeds)
        bad_meshes_out      = self.apply_stacked_mesh_gru(bad_meshes_embeds)
        #
        good_out_pp         = torch.cat([good_out, doc_gaf, good_meshes_out], -1)
        bad_out_pp          = torch.cat([bad_out,  doc_baf, bad_meshes_out],  -1)
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


docs            = '/home/dpappas/for_ryan/test_batch_1/bioasq6_bm25_top100/bioasq6_bm25_docset_top100.test.pkl'
retrieved       = '/home/dpappas/for_ryan/test_batch_1/bioasq6_bm25_top100/bioasq6_bm25_top100.test.pkl'
golden          = '/home/dpappas/for_ryan/test_batch_1/bioasq6_bm25_top100/bioasq6_bm25_top100.test.golden.pkl'

test_data, test_docs, idf, max_idf, wv = load_all_data(w2v_bin_path, idf_pickle_path)

odir            = '/home/dpappas/test_model_1/'
resume_from     = '/home/dpappas/model_1_run0/best_checkpoint.pth.tar'

model           = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool, k_sent_maxpool=k_sent_maxpool)
params          = model.parameters()

load_model_from_checkpoint(resume_from)


for quer in test_data[u'queries']:
    qid                         = quer['query_id']
    qtext                       = quer['query_text']
    quest_tokens, quest_embeds  = get_embeds(tokenize(qtext), wv)
    q_idfs                      = np.array([[idf_val(qw)] for qw in quest_tokens], 'float')
    for retr_doc in quer['retrieved_documents']:
        doc_id      = retr_doc['doc_id']
        bm25        = retr_doc['norm_bm25_score']
        the_doc     = test_docs[doc_id]
        is_relevant = retr_doc['is_relevant']
        (
            good_sents_embeds, good_sents_escores, good_doc_af,
            good_meshes_embeds, held_out_sents
        ) = prep_data(qtext, the_doc, bm25)
        doc_emit_, gs_emits_    = model.emit_one(
            doc1_sents_embeds   = good_sents_embeds,
            question_embeds     = quest_embeds,
            q_idfs              = q_idfs,
            sents_gaf           = good_sents_escores,
            doc_gaf             = good_doc_af,
            good_meshes_embeds  = good_meshes_embeds
        )
        emition     = doc_emit_.cpu().item()
        emitss      = gs_emits_.tolist()
        mmax        = max(emitss)
        print(emition, is_relevant)
        print(emitss)
        # all_emits, extracted_from_one = [], []
        # for ind in range(len(emitss)):
        #     t = (
        #         snip_is_relevant(held_out_sents[ind], gold_snips),
        #         emitss[ind],
        #         "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(retr['doc_id']),
        #         held_out_sents[ind]
        #     )
        #     all_emits.append(t)
        #     if (emitss[ind] == mmax):
        #         extracted_from_one.append(t)
        # doc_res[retr['doc_id']] = float(emition)
        # all_emits = sorted(all_emits, key=lambda x: x[1], reverse=True)