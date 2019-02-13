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

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
softmax     = lambda z: np.exp(z) / np.sum(np.exp(z))
stopwords   = nltk.corpus.stopwords.words("english")

def get_bm25_metrics(avgdl=0., mean=0., deviation=0.):
    if(avgdl == 0):
        total_words = 0
        total_docs  = 0
        for dic in tqdm(train_docs):
            sents = sent_tokenize(train_docs[dic]['title']) + sent_tokenize(train_docs[dic]['abstractText'])
            for s in sents:
                total_words += len(tokenize(s))
                total_docs  += 1.
        avgdl = float(total_words) / float(total_docs)
    else:
        print('avgdl {} provided'.format(avgdl))
    #
    if(mean == 0 and deviation==0):
        BM25scores  = []
        k1, b       = 1.2, 0.75
        not_found   = 0
        for qid in tqdm(bioasq6_data):
            qtext           = bioasq6_data[qid]['body']
            all_retr_ids    = [link.split('/')[-1] for link in bioasq6_data[qid]['documents']]
            for dic in all_retr_ids:
                try:
                    sents   = sent_tokenize(train_docs[dic]['title']) + sent_tokenize(train_docs[dic]['abstractText'])
                    q_toks  = tokenize(qtext)
                    for sent in sents:
                        BM25score = similarity_score(q_toks, tokenize(sent), k1, b, idf, avgdl, False, 0, 0, max_idf)
                        BM25scores.append(BM25score)
                except KeyError:
                    not_found += 1
        #
        mean        = sum(BM25scores)/float(len(BM25scores))
        nominator   = 0
        for score in BM25scores:
            nominator += ((score - mean) ** 2)
        deviation   = math.sqrt((nominator) / float(len(BM25scores) - 1))
    else:
        print('mean {} provided'.format(mean))
        print('deviation {} provided'.format(deviation))
    return avgdl, mean, deviation

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

# Compute the average length from a collection of documents
def compute_avgdl(documents):
    total_words = 0
    for document in documents:
        total_words += len(document)
    avgdl = total_words / len(documents)
    return avgdl

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))

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

def get_snippets_loss(gs_emits_, bs_emits_):
    if(len(gs_emits_)>0):
        gs_emits_       = torch.cat([t[0].unsqueeze(-1) for t in gs_emits_])
        gs_emits_       = gs_emits_.squeeze(-1)
        tags_1          = torch.ones_like(gs_emits_)
        if(use_cuda):
            tags_1 = tags_1.cuda()
        sn_d1_l = F.binary_cross_entropy(gs_emits_, tags_1, reduction='sum')
    else:
        sn_d1_l = 0.0
        #
    if(len(bs_emits_)>0):
        bs_emits_       = torch.cat([t[0].unsqueeze(-1) for t in bs_emits_])
        bs_emits_       = bs_emits_.squeeze(-1)
        tags_2          = torch.zeros_like(bs_emits_)
        if(use_cuda):
            tags_2 = tags_2.cuda()
        sn_d2_l         = F.binary_cross_entropy(bs_emits_, tags_2, reduction='sum')
    else:
        sn_d2_l = 0.0
    return sn_d1_l + sn_d2_l

def get_doc_loss(pos_score, many_neg_scores):
    ret     = 0.0
    target  = autograd.Variable(torch.FloatTensor([1]), requires_grad=False)
    if(use_cuda):
        target = target.cuda()
    for neg_score in many_neg_scores:
        hl      = model.margin_loss(pos_score.unsqueeze(0), neg_score.unsqueeze(0), target)
        ret     += hl
    return ret

def init_the_logger(hdlr):
    if not os.path.exists(odir):
        os.makedirs(odir)
    od          = odir.split('/')[-1] # 'sent_posit_drmm_MarginRankingLoss_0p001'
    logger      = logging.getLogger(od)
    if(hdlr is not None):
        logger.removeHandler(hdlr)
    hdlr        = logging.FileHandler(os.path.join(odir,'model.log'))
    formatter   = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr

def get_words(s, idf, max_idf):
    sl  = tokenize(s)
    sl  = [s for s in sl]
    sl2 = [s for s in sl if idf_val(s, idf, max_idf) >= 2.0]
    return sl, sl2

def tokenize(x):
  return bioclean(x)

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

def get_embeds_use_unk(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if(tok in wv):
            ret1.append(tok)
            ret2.append(wv[tok])
        else:
            wv[tok] = np.random.randn(embedding_dim)
            ret1.append(tok)
            ret2.append(wv[tok])
    return ret1, np.array(ret2, 'float64')

def get_embeds_use_only_unk(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        wv[tok] = np.random.randn(embedding_dim)
        ret1.append(tok)
        ret2.append(wv[tok])
    return ret1, np.array(ret2, 'float64')

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

def get_gold_snips(quest_id, bioasq6_data):
    gold_snips                  = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            gold_snips.extend(sent_tokenize(sn['text'].strip()))
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
            ind_from                            = docs[pid]['title'].index(the_text)
            ind_to                              = ind_from + len(the_text)
            esnip_res["beginSection"]           = "title"
            esnip_res["endSection"]             = "title"
            esnip_res["offsetInBeginSection"]   = ind_from
            esnip_res["offsetInEndSection"]     = ind_to
        except:
            # print(the_text)
            # pprint(docs[pid])
            ind_from                            = docs[pid]['abstractText'].index(the_text)
            ind_to                              = ind_from + len(the_text)
            esnip_res["beginSection"]           = "abstract"
            esnip_res["endSection"]             = "abstract"
            esnip_res["offsetInBeginSection"]   = ind_from
            esnip_res["offsetInEndSection"]     = ind_to
        ret['snippets'].append(esnip_res)
    return ret

def get_snips(quest_id, gid, bioasq6_data):
    good_snips = []
    if('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            if(sn['document'].endswith(gid)):
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
    good_mesh = ['mesh'] + good_mesh
    # good_mesh = ' # '.join(good_mesh)
    # good_mesh = good_mesh.split()
    # good_mesh = [gm.split() for gm in good_mesh]
    good_mesh = [gm for gm in good_mesh]
    return good_mesh

def snip_is_relevant(one_sent, gold_snips):
    # print one_sent
    # pprint(gold_snips)
    one_sent    = ' '.join(bioclean(one_sent)).strip()
    gold_snips  = [' '.join(bioclean(t)).strip() for t in gold_snips]
    return int(
        any([(one_sent in gold_snip) or (gold_snip in one_sent) for gold_snip in gold_snips])
    )

def train_one(epoch, bioasq6_data):
    model.train()
    batch_costs, batch_acc, epoch_costs, epoch_acc = [], [], [], []
    batch_counter, epoch_aver_cost, epoch_aver_acc = 0, 0., 0.
    #
    start_time = time.time()
    random.shuffle(train_data['queries'])
    pbar = tqdm(train_data['queries'])
    for datum in pbar:
        # get querry id and text
        qid, qtext                      = datum['query_id'], datum['query_text']
        # get the embeds and idf of querry tokens
        quest_tokens, quest_embeds      = get_embeds(tokenize(qtext), wv)
        q_idfs                          = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
        # ignore if you cannot find an retrieved_document or any relevant snippet overall
        if(len(datum['retrieved_documents']) == 0 or 'snippets' not in bioasq6_data[qid]):
            continue
        # ignore if you cannot find a retrieved_document or any relevant snippet
        found_rel_snippet_flag          = False
        doc_results, sent_results       = [], []
        for retr_doc in datum['retrieved_documents']:
            pmid        = retr_doc['doc_id']
            the_bm25    = retr_doc['norm_bm25_score']
            is_relevant = retr_doc['is_relevant']
            ##########
            title       = train_docs[pmid]['title']
            abs         = train_docs[pmid]['abstractText']
            all_sents   = [title] + sent_tokenize(abs)
            ##########
            gold_snips  = []
            for snip in bioasq6_data[qid]['snippets']:
                if (snip['document'].endswith(pmid)):
                    gold_snips.extend([s.strip() for s in sent_tokenize(snip['text'].strip())])
            ##########
            doc_af      = GetScores(qtext, ' '.join(all_sents), the_bm25, idf, max_idf)
            doc_af.append(len(all_sents) / 60.)
            doc_af.append(len(' '.join(all_sents)) / 800.)
            sents_embeds, sents_escores, held_out_sents, sent_tags = [], [], [], []
            for sent in all_sents:
                sent_toks                = tokenize(sent)
                good_tokens, good_embeds = get_embeds(sent_toks, wv)
                good_escores = GetScores(qtext, sent, the_bm25, idf, max_idf)[:-1]
                good_escores.append(len(sent_toks) / 342.)
                if (len(good_tokens) > 0):
                    sents_embeds.append(good_embeds)
                    sents_escores.append(good_escores)
                    held_out_sents.append(sent)
                    sent_tags.append(snip_is_relevant(sent, gold_snips))
            if(len(gold_snips)>0):
                found_rel_snippet_flag = True
                # pprint(gold_snips)
                # pprint(list(zip(held_out_sents, sent_tags)))
            ##########
            doc_emit_, gs_emits_    = model.emit_one(
                doc1_sents_embeds   = sents_embeds,
                question_embeds     = quest_embeds,
                q_idfs              = q_idfs,
                sents_gaf           = sents_escores,
                doc_gaf             = doc_af
            )
            ##########
            doc_results.append((doc_emit_ , is_relevant))
            sent_results.extend(list(zip(gs_emits_, sent_tags)))
        # ignore if you cannot find any relevant snippets in the retrieved documents
        if not found_rel_snippet_flag:
            continue
        # sort the document results
        doc_results         = sorted([t for t in doc_results], key=lambda x: x[0], reverse=True)
        # sort the positive snippets results
        good_sent_results   = sorted([t for t in sent_results if(t[1]==1)], key=lambda x: x[0], reverse=True)
        # sort the negative snippets results
        bad_sent_results    = sorted([t for t in sent_results if(t[1]==0)], key=lambda x: x[0], reverse=True)
        bad_sent_results    = bad_sent_results[:2*len(good_sent_results)]
        # compute the doc_ret loss and the snip_extr loss
        doc_loss            = 0.0
        for i in range(len(doc_results)):
            if(doc_results[i][1]):
                pos_score       = doc_results[i][0]
                many_neg_scores = [doc_results[j][0] for j in range(i) if(not doc_results[j][1])]
                doc_loss        += get_doc_loss(pos_score, many_neg_scores)
        snip_loss               = get_snippets_loss(good_sent_results, bad_sent_results)
        # add all losses to one and add the result to batch losses and epoch losses
        total_loss              = 0.5 * doc_loss + 0.5 * snip_loss
        batch_costs.append(total_loss)
        epoch_costs.append(total_loss)
        # if reached batch size then back propagate
        if (len(batch_costs) == b_size):
            batch_counter   += 1
            batch_cost      = sum(batch_costs) / float(len(batch_costs))
            batch_cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            elapsed_time    = time.time() - start_time
            start_time      = time.time()
            batch_costs     = []
            batch_acc       = []
            aver_epoch_cost = sum(epoch_costs) / float(len(epoch_costs))
            aver_epoch_cost = aver_epoch_cost.cpu().item()
            pbar.set_description('{} || {}'.format(batch_cost.cpu().item(), aver_epoch_cost))
    if (len(batch_costs) > 0):
        batch_counter += 1
        batch_cost = sum(batch_costs) / float(len(batch_costs))
        batch_cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        elapsed_time = time.time() - start_time
        start_time = time.time()
        batch_costs = []
        batch_acc = []
        aver_epoch_cost = sum(epoch_costs) / float(len(epoch_costs))
        aver_epoch_cost = aver_epoch_cost.cpu().item()
        pbar.set_description('{} || {}'.format(batch_cost.cpu().item(), aver_epoch_cost))
        print('Done epoch {}'.format(epoch))

def do_for_one_retrieved(doc_emit_, gs_emits_, held_out_sents, retr, doc_res, gold_snips):
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
    all_emits               = sorted(all_emits, key=lambda x: x[1], reverse=True)
    return doc_res, extracted_from_one, all_emits

def get_norm_doc_scores(the_doc_scores):
    ks = list(the_doc_scores.keys())
    vs = [the_doc_scores[k] for k in ks]
    vs = softmax(vs)
    norm_doc_scores = {}
    for i in range(len(ks)):
        norm_doc_scores[ks[i]] = vs[i]
    return norm_doc_scores

def select_snippets_v1(extracted_snippets):
    '''
    :param extracted_snippets:
    :param doc_res:
    :return: returns the best 10 snippets of all docs (0..n from each doc)
    '''
    sorted_snips = sorted(extracted_snippets, key=lambda x: x[1], reverse=True)
    return sorted_snips[:10]

def select_snippets_v2(extracted_snippets):
    '''
    :param extracted_snippets:
    :param doc_res:
    :return: returns the best snippet of each doc  (1 from each doc)
    '''
    # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
    ret                 = {}
    for es in extracted_snippets:
        if(es[2] in ret):
            if(es[1] > ret[es[2]][1]):
                ret[es[2]] = es
        else:
            ret[es[2]] = es
    sorted_snips =  sorted(ret.values(), key=lambda x: x[1], reverse=True)
    return sorted_snips[:10]

def select_snippets_v3(extracted_snippets, the_doc_scores):
    '''
    :param      extracted_snippets:
    :param      doc_res:
    :return:    returns the top 10 snippets across all documents (0..n from each doc)
    '''
    norm_doc_scores     = get_norm_doc_scores(the_doc_scores)
    # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
    extracted_snippets  = [tt for tt in extracted_snippets if (tt[2] in norm_doc_scores)]
    sorted_snips        = sorted(extracted_snippets, key=lambda x: x[1] * norm_doc_scores[x[2]], reverse=True)
    return sorted_snips[:10]

def prep_data(quest, the_doc, the_bm25, wv, good_snips, idf, max_idf, use_sent_tokenizer):
    good_sents      = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
    ####
    good_doc_af     = GetScores(quest, ' '.join(good_sents), the_bm25, idf, max_idf)
    good_doc_af.append(len(good_sents) / 60.)
    good_doc_af.append(len(' '.join(good_sents)) / 800.)
    ####
    good_sents_embeds, good_sents_escores, held_out_sents, good_sent_tags = [], [], [], []
    for good_text in good_sents:
        sent_toks                   = tokenize(good_text)
        good_tokens, good_embeds    = get_embeds(sent_toks, wv)
        good_escores                = GetScores(quest, good_text, the_bm25, idf, max_idf)[:-1]
        good_escores.append(len(sent_toks)/ 342.)
        if (len(good_embeds) > 0):
            good_sents_embeds.append(good_embeds)
            good_sents_escores.append(good_escores)
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

def do_for_some_retrieved(docs, dato, retr_docs, data_for_revision, ret_data, use_sent_tokenizer):
    emitions                    = {
        'body': dato['query_text'],
        'id': dato['query_id'],
        'documents': []
    }
    #
    quest_text                  = dato['query_text']
    quest_tokens, quest_embeds  = get_embeds(tokenize(quest_text), wv)
    q_idfs                      = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
    gold_snips                  = get_gold_snips(dato['query_id'], bioasq6_data)
    #
    doc_res, extracted_snippets         = {}, []
    extracted_snippets_known_rel_num    = []
    for retr in retr_docs:
        datum                   = prep_data(quest_text, docs[retr['doc_id']], retr['norm_bm25_score'], wv, gold_snips, idf, max_idf, use_sent_tokenizer=use_sent_tokenizer)
        doc_emit_, gs_emits_    = model.emit_one(
            doc1_sents_embeds   = datum['sents_embeds'],
            question_embeds     = quest_embeds,
            q_idfs              = q_idfs,
            sents_gaf           = datum['sents_escores'],
            doc_gaf             = datum['doc_af']
        )
        doc_res, extracted_from_one, all_emits = do_for_one_retrieved(doc_emit_, gs_emits_, datum['held_out_sents'], retr, doc_res, gold_snips)
        # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
        extracted_snippets.extend(extracted_from_one)
        #
        total_relevant = sum([1 for em in all_emits if(em[0]==True)])
        if (total_relevant > 0):
            extracted_snippets_known_rel_num.extend(all_emits[:total_relevant])
        if (dato['query_id'] not in data_for_revision):
            data_for_revision[dato['query_id']] = {'query_text': dato['query_text'], 'snippets'  : {retr['doc_id']: all_emits}}
        else:
            data_for_revision[dato['query_id']]['snippets'][retr['doc_id']] = all_emits
    #
    doc_res                                 = sorted(doc_res.items(), key=lambda x: x[1], reverse=True)
    the_doc_scores                          = dict([("http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]), pm[1]) for pm in doc_res[:10]])
    doc_res                                 = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
    emitions['documents']                   = doc_res[:100]
    ret_data['questions'].append(emitions)
    #
    extracted_snippets                      = [tt for tt in extracted_snippets if (tt[2] in doc_res[:10])]
    extracted_snippets_known_rel_num        = [tt for tt in extracted_snippets_known_rel_num if (tt[2] in doc_res[:10])]
    if(use_sent_tokenizer):
        extracted_snippets_v1               = select_snippets_v1(extracted_snippets)
        extracted_snippets_v2               = select_snippets_v2(extracted_snippets)
        extracted_snippets_v3               = select_snippets_v3(extracted_snippets, the_doc_scores)
        extracted_snippets_known_rel_num_v1 = select_snippets_v1(extracted_snippets_known_rel_num)
        extracted_snippets_known_rel_num_v2 = select_snippets_v2(extracted_snippets_known_rel_num)
        extracted_snippets_known_rel_num_v3 = select_snippets_v3(extracted_snippets_known_rel_num, the_doc_scores)
    else:
        extracted_snippets_v1, extracted_snippets_v2, extracted_snippets_v3 = [], [], []
        extracted_snippets_known_rel_num_v1, extracted_snippets_known_rel_num_v2, extracted_snippets_known_rel_num_v3 = [], [], []
    #
    # pprint(extracted_snippets_v1)
    # pprint(extracted_snippets_v2)
    # pprint(extracted_snippets_v3)
    # exit()
    snips_res_v1                = prep_extracted_snippets(extracted_snippets_v1, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    snips_res_v2                = prep_extracted_snippets(extracted_snippets_v2, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    snips_res_v3                = prep_extracted_snippets(extracted_snippets_v3, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    # pprint(snips_res_v1)
    # pprint(snips_res_v2)
    # pprint(snips_res_v3)
    # exit()
    #
    snips_res_known_rel_num_v1  = prep_extracted_snippets(extracted_snippets_known_rel_num_v1, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    snips_res_known_rel_num_v2  = prep_extracted_snippets(extracted_snippets_known_rel_num_v2, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    snips_res_known_rel_num_v3  = prep_extracted_snippets(extracted_snippets_known_rel_num_v3, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    #
    snips_res = {
        'v1' : snips_res_v1,
        'v2' : snips_res_v2,
        'v3' : snips_res_v3,
    }
    snips_res_known = {
        'v1' : snips_res_known_rel_num_v1,
        'v2' : snips_res_known_rel_num_v2,
        'v3' : snips_res_known_rel_num_v3,
    }
    return data_for_revision, ret_data, snips_res, snips_res_known

def print_the_results(prefix, all_bioasq_gold_data, all_bioasq_subm_data, all_bioasq_subm_data_known, data_for_revision):
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data_known, data_for_revision)
    pprint(bioasq_snip_res)
    print('{} known MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    print('{} known F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    print('{} known MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    print('{} known GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    logger.info('{} known MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    logger.info('{} known F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    logger.info('{} known MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    logger.info('{} known GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data, data_for_revision)
    pprint(bioasq_snip_res)
    print('{} MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    print('{} F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    print('{} MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    print('{} GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    logger.info('{} MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    logger.info('{} F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    logger.info('{} MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    logger.info('{} GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #

def get_one_map(prefix, data, docs, use_sent_tokenizer):
    model.eval()
    #
    ret_data                        = {'questions': []}
    all_bioasq_subm_data_v1         = {"questions": []}
    all_bioasq_subm_data_known_v1   = {"questions": []}
    all_bioasq_subm_data_v2         = {"questions": []}
    all_bioasq_subm_data_known_v2   = {"questions": []}
    all_bioasq_subm_data_v3         = {"questions": []}
    all_bioasq_subm_data_known_v3   = {"questions": []}
    all_bioasq_gold_data            = {'questions': []}
    data_for_revision               = {}
    #
    for dato in tqdm(data['queries']):
        all_bioasq_gold_data['questions'].append(bioasq6_data[dato['query_id']])
        data_for_revision, ret_data, snips_res, snips_res_known = do_for_some_retrieved(docs, dato, dato['retrieved_documents'], data_for_revision, ret_data, use_sent_tokenizer)
        all_bioasq_subm_data_v1['questions'].append(snips_res['v1'])
        all_bioasq_subm_data_v2['questions'].append(snips_res['v2'])
        all_bioasq_subm_data_v3['questions'].append(snips_res['v3'])
        all_bioasq_subm_data_known_v1['questions'].append(snips_res_known['v1'])
        all_bioasq_subm_data_known_v2['questions'].append(snips_res_known['v3'])
        all_bioasq_subm_data_known_v3['questions'].append(snips_res_known['v3'])
    #
    print_the_results('v1 '+prefix, all_bioasq_gold_data, all_bioasq_subm_data_v1, all_bioasq_subm_data_known_v1, data_for_revision)
    print_the_results('v2 '+prefix, all_bioasq_gold_data, all_bioasq_subm_data_v2, all_bioasq_subm_data_known_v2, data_for_revision)
    print_the_results('v3 '+prefix, all_bioasq_gold_data, all_bioasq_subm_data_v3, all_bioasq_subm_data_known_v3, data_for_revision)
    #
    if (prefix == 'dev'):
        with open(os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(dataloc+'bioasq.dev.json', os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json'))
    else:
        with open(os.path.join(odir,'elk_relevant_abs_posit_drmm_lists_test.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(dataloc+'bioasq.test.json', os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_test.json'))
    return res_map

def load_all_data(dataloc, w2v_bin_path, idf_pickle_path):
    print('loading pickle data')
    #
    with open(dataloc+'BioASQ-trainingDataset6b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict( (q['id'], q) for q in bioasq6_data['questions'] )
    #
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
    #
    print('loading idfs')
    idf, max_idf    = load_idfs(idf_pickle_path, words)
    print('loading w2v')
    wv              = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    wv              = dict([(word, wv[word]) for word in wv.vocab.keys() if(word in words)])
    return test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, wv, bioasq6_data

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
        self.doc_add_feats                          = 6
        self.sent_add_feats                         = 4
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
        res = []
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
        return ret, res
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
    def emit_one(self, doc1_sents_embeds, question_embeds, q_idfs, sents_gaf, doc_gaf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        if(use_cuda):
            q_idfs          = q_idfs.cuda()
            question_embeds = question_embeds.cuda()
            doc_gaf         = doc_gaf.cuda()
        #
        q_context       = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context       = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits  = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights, self.k_sent_maxpool)
        #
        good_out_pp         = torch.cat([good_out, doc_gaf], -1)
        #
        final_good_output   = self.final_layer_1(good_out_pp)
        final_good_output   = self.final_activ_1(final_good_output)
        final_good_output   = self.final_layer_2(final_good_output)
        #
        gs_emits            = gs_emits.unsqueeze(-1)
        gs_emits            = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits            = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits            = torch.sigmoid(gs_emits)
        #
        return final_good_output, gs_emits
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, question_embeds, q_idfs, sents_gaf, sents_baf, doc_gaf, doc_baf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        doc_baf             = autograd.Variable(torch.FloatTensor(doc_baf),             requires_grad=False)
        if(use_cuda):
            q_idfs          = q_idfs.cuda()
            question_embeds = question_embeds.cuda()
            doc_gaf         = doc_gaf.cuda()
            doc_baf         = doc_baf.cuda()
        #
        q_context           = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context           = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits  = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights, self.k_sent_maxpool)
        bad_out, bs_emits   = self.do_for_one_doc_cnn(doc2_sents_embeds, sents_baf, question_embeds, q_context, q_weights, self.k_sent_maxpool)
        #
        good_out_pp         = torch.cat([good_out, doc_gaf], -1)
        bad_out_pp          = torch.cat([bad_out, doc_baf], -1)
        #
        final_good_output   = self.final_layer_1(good_out_pp)
        final_good_output   = self.final_activ_1(final_good_output)
        final_good_output   = self.final_layer_2(final_good_output)
        #
        gs_emits            = gs_emits.unsqueeze(-1)
        gs_emits            = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits            = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits            = torch.sigmoid(gs_emits)
        #
        final_bad_output    = self.final_layer_1(bad_out_pp)
        final_bad_output    = self.final_activ_1(final_bad_output)
        final_bad_output    = self.final_layer_2(final_bad_output)
        #
        bs_emits            = bs_emits.unsqueeze(-1)
        bs_emits            = torch.cat([bs_emits, final_good_output.unsqueeze(-1).expand_as(bs_emits)], -1)
        bs_emits            = self.oo_layer(bs_emits).squeeze(-1)
        bs_emits            = torch.sigmoid(bs_emits)
        #
        loss1               = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output, gs_emits, bs_emits

use_cuda = torch.cuda.is_available()

# # laptop
# w2v_bin_path        = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/dpappas/for_ryan/fordp/idf.pkl'
# dataloc             = '/home/dpappas/for_ryan/'
# eval_path           = '/home/dpappas/for_ryan/eval/run_eval.py'
# retrieval_jar_path  = '/home/dpappas/NetBeansProjects/my_bioasq_eval_2/dist/my_bioasq_eval_2.jar'
# # use_cuda            = False
# odd                 = '/home/dpappas/'
# # get_embeds          = get_embeds_use_unk
# # get_embeds          = get_embeds_use_only_unk

# # cslab241
# w2v_bin_path        = '/home/dpappas/for_ryan/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/dpappas/for_ryan/idf.pkl'
# dataloc             = '/home/DATA/Biomedical/document_ranking/bioasq_data/'
# eval_path           = '/home/DATA/Biomedical/document_ranking/eval/run_eval.py'
# retrieval_jar_path  = '/home/dpappas/bioasq_eval/dist/my_bioasq_eval_2.jar'
# odd                 = '/home/dpappas/'

# atlas , cslab243
w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
dataloc             = '/home/dpappas/bioasq_all/bioasq_data/'
eval_path           = '/home/dpappas/bioasq_all/eval/run_eval.py'
retrieval_jar_path  = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'
odd                 = '/home/dpappas/'
use_cuda            = True
# get_embeds          = get_embeds_use_unk
# get_embeds          = get_embeds_use_only_unk

# # gpu_server_1
# w2v_bin_path        = '/media/large_space_1/DATA/bioasq_all/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/media/large_space_1/DATA/bioasq_all/idf.pkl'
# dataloc             = '/media/large_space_1/DATA/bioasq_all/bioasq_data/'
# eval_path           = '/media/large_space_1/DATA/bioasq_all/eval/run_eval.py'
# retrieval_jar_path  = '/media/large_space_1/DATA/bioasq_all/dist/my_bioasq_eval_2.jar'
# odd                 = '/media/large_space_1/dpappas/'

# # gpu_server_2
# w2v_bin_path        = '/home/cave/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/cave/dpappas/bioasq_all/idf.pkl'
# dataloc             = '/home/cave/dpappas/bioasq_all/bioasq_data/'
# eval_path           = '/home/cave/dpappas/bioasq_all/eval/run_eval.py'
# retrieval_jar_path  = '/home/cave/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'
# odd                 = '/home/cave/dpappas/'

k_for_maxpool       = 5
k_sent_maxpool      = 5
embedding_dim       = 30 #200
lr                  = 0.01
b_size              = 32
max_epoch           = 10

hdlr = None
for run in range(0, 5):
    #
    my_seed = run
    random.seed(my_seed)
    torch.manual_seed(my_seed)
    #0
    odir = 'sigir_joint_classic_myfeed_run_{}/'.format(run)
    odir    = os.path.join(odd, odir)
    print(odir)
    if(not os.path.exists(odir)):
        os.makedirs(odir)
    #
    logger, hdlr    = init_the_logger(hdlr)
    print('random seed: {}'.format(my_seed))
    logger.info('random seed: {}'.format(my_seed))
    #
    (
        test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, wv, bioasq6_data
    ) = load_all_data(dataloc=dataloc, w2v_bin_path=w2v_bin_path, idf_pickle_path=idf_pickle_path)
    #
    # avgdl, mean, deviation = get_bm25_metrics(avgdl=21.2508, mean=0.5973, deviation=0.5926)
    # print(avgdl, mean, deviation)
    #
    print('Compiling model...')
    logger.info('Compiling model...')
    model       = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool)
    if(use_cuda):
        model   = model.cuda()
    params      = model.parameters()
    print_params(model)
    optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #
    best_dev_map, test_map = None, None
    for epoch in range(max_epoch):
        train_one(epoch+1, bioasq6_data)
        epoch_dev_map       = get_one_map('dev', dev_data, dev_docs, use_sent_tokenizer=True)
        if(best_dev_map is None or epoch_dev_map>=best_dev_map):
            best_dev_map    = epoch_dev_map
            test_map        = get_one_map('test', test_data, test_docs, use_sent_tokenizer=True)
            save_checkpoint(epoch, model, best_dev_map, optimizer, filename=os.path.join(odir,'best_checkpoint.pth.tar'))
        print('epoch:{:02d} epoch_dev_map:{:.4f} best_dev_map:{:.4f} test_map:{:.4f}'.format(epoch + 1, epoch_dev_map, best_dev_map, test_map))
        logger.info('epoch:{:02d} epoch_dev_map:{:.4f} best_dev_map:{:.4f} test_map:{:.4f}'.format(epoch + 1, epoch_dev_map, best_dev_map, test_map))

'''
29,041,002 articles
29,017,626 with title
18,707,863 with abstract as well 

Is Hirschsprung disease a mendelian or a multifactorial disorder

search: _exists_:AbstractText AND AbstractText:/.+/ AND _exists_:ArticleTitle AND ArticleTitle:/.+/


'''
