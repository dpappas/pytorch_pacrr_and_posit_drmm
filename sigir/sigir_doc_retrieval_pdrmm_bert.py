#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

import os
import json
import time
import random
import logging
import subprocess
import torch
import torch.nn.functional         as F
import torch.nn                    as nn
import numpy                       as np
import torch.optim                 as optim
# import  cPickle                     as pickle
import pickle
import torch.autograd              as autograd
from tqdm import tqdm
from pprint import pprint
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher
import re
import nltk
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pytorch_pretrained_bert.tokenization import BertTokenizer

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                            t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                          '').strip().lower()).split()
softmax = lambda z: np.exp(z) / np.sum(np.exp(z))
stopwords = nltk.corpus.stopwords.words("english")


def get_bm25_metrics(avgdl=0., mean=0., deviation=0.):
    if (avgdl == 0):
        total_words = 0
        total_docs = 0
        for dic in tqdm(train_docs, ascii=True):
            sents = sent_tokenize(train_docs[dic]['title']) + sent_tokenize(train_docs[dic]['abstractText'])
            for s in sents:
                total_words += len(tokenize(s))
                total_docs += 1.
        avgdl = float(total_words) / float(total_docs)
    else:
        print('avgdl {} provided'.format(avgdl))
    #
    if (mean == 0 and deviation == 0):
        BM25scores = []
        k1, b = 1.2, 0.75
        not_found = 0
        for qid in tqdm(bioasq6_data, ascii=True):
            qtext = bioasq6_data[qid]['body']
            all_retr_ids = [link.split('/')[-1] for link in bioasq6_data[qid]['documents']]
            for dic in all_retr_ids:
                try:
                    sents = sent_tokenize(train_docs[dic]['title']) + sent_tokenize(train_docs[dic]['abstractText'])
                    q_toks = tokenize(qtext)
                    for sent in sents:
                        BM25score = similarity_score(q_toks, tokenize(sent), k1, b, idf, avgdl, False, 0, 0, max_idf)
                        BM25scores.append(BM25score)
                except KeyError:
                    not_found += 1
        #
        mean = sum(BM25scores) / float(len(BM25scores))
        nominator = 0
        for score in BM25scores:
            nominator += ((score - mean) ** 2)
        deviation = math.sqrt((nominator) / float(len(BM25scores) - 1))
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
        return tf / len(document)


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
            score += idf_scores[query_term] * ((tf(query_term, document) * (k1 + 1)) / (
                    tf(query_term, document) + k1 * (1 - b + b * (len(document) / avgdl))))
    if normalize:
        return ((score - mean) / deviation)
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
            doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
            year = doc_text[doc_id]['publicationDate'].split('-')[0]
            ##########################
            # Skip 2017/2018 docs always. Skip 2016 docs for training.
            # Need to change for final model - 2017 should be a train year only.
            # Use only for testing.
            if year == '2017' or year == '2018' or (train and year == '2016'):
                # if year == '2018' or (train and year == '2017'):
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
    trainable = 0
    untrainable = 0
    for parameter in model.parameters():
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        if (parameter.requires_grad):
            trainable += v
        else:
            untrainable += v
    total_params = trainable + untrainable
    print(40 * '=')
    print('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    print(40 * '=')
    logger.info(40 * '=')
    logger.info('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    logger.info(40 * '=')


def compute_the_cost(costs, back_prop=True):
    cost_ = torch.stack(costs)
    cost_ = cost_.sum() / (1.0 * cost_.size(0))
    if (back_prop):
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
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_valid_score': max_dev_map,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)


def get_map_res(fgold, femit):
    trec_eval_res = subprocess.Popen(['python', eval_path, fgold, femit], stdout=subprocess.PIPE, shell=False)
    (out, err) = trec_eval_res.communicate()
    lines = out.decode("utf-8").split('\n')
    map_res = [l for l in lines if (l.startswith('map '))][0].split('\t')
    map_res = float(map_res[-1])
    return map_res


def back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc):
    batch_cost = sum(batch_costs) / float(len(batch_costs))
    batch_cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    batch_aver_cost = batch_cost.cpu().item()
    epoch_aver_cost = sum(epoch_costs) / float(len(epoch_costs))
    batch_aver_acc = sum(batch_acc) / float(len(batch_acc))
    epoch_aver_acc = sum(epoch_acc) / float(len(epoch_acc))
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
    fgold = '{}_data_for_revision.json'.format(prefix)
    fgold = os.path.join(odir, fgold)
    fgold = os.path.abspath(fgold)
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
    fgold = '{}_gold_bioasq.json'.format(prefix)
    fgold = os.path.join(odir, fgold)
    fgold = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_gold, indent=4, sort_keys=True))
        f.close()
    #
    femit = '{}_emit_bioasq.json'.format(prefix)
    femit = os.path.join(odir, femit)
    femit = os.path.abspath(femit)
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
    (out, err) = bioasq_eval_res.communicate()
    lines = out.decode("utf-8").split('\n')
    ret = {}
    for line in lines:
        if (':' in line):
            k = line.split(':')[0].strip()
            v = line.split(':')[1].strip()
            ret[k] = float(v)
    return ret


def similar(upstream_seq, downstream_seq):
    upstream_seq = upstream_seq.encode('ascii', 'ignore')
    downstream_seq = downstream_seq.encode('ascii', 'ignore')
    s = SequenceMatcher(None, upstream_seq, downstream_seq)
    match = s.find_longest_match(0, len(upstream_seq), 0, len(downstream_seq))
    upstream_start = match[0]
    upstream_end = match[0] + match[2]
    longest_match = upstream_seq[upstream_start:upstream_end]
    to_match = upstream_seq if (len(downstream_seq) > len(upstream_seq)) else downstream_seq
    r1 = SequenceMatcher(None, to_match, longest_match).ratio()
    return r1


def get_pseudo_retrieved(dato):
    some_ids = [item['document'].split('/')[-1].strip() for item in bioasq6_data[dato['query_id']]['snippets']]
    pseudo_retrieved = [
        {
            'bm25_score': 7.76,
            'doc_id': id,
            'is_relevant': True,
            'norm_bm25_score': 3.85
        }
        for id in set(some_ids)
    ]
    return pseudo_retrieved


def get_snippets_loss(good_sent_tags, gs_emits_, bs_emits_):
    wright = torch.cat([gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 1)])
    wrong = [gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 0)]
    wrong = torch.cat(wrong + [bs_emits_.squeeze(-1)])
    losses = [model.my_hinge_loss(w.unsqueeze(0).expand_as(wrong), wrong) for w in wright]
    return sum(losses) / float(len(losses))


def get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_):
    bs_emits_ = bs_emits_.squeeze(-1)
    gs_emits_ = gs_emits_.squeeze(-1)
    good_sent_tags = torch.FloatTensor(good_sent_tags)
    tags_2 = torch.zeros_like(bs_emits_)
    if (use_cuda):
        good_sent_tags = good_sent_tags.cuda()
        tags_2 = tags_2.cuda()
    #
    # sn_d1_l = F.binary_cross_entropy(gs_emits_, good_sent_tags, size_average=False, reduce=True)
    # sn_d2_l = F.binary_cross_entropy(bs_emits_, tags_2, size_average=False, reduce=True)
    sn_d1_l = F.binary_cross_entropy(gs_emits_, good_sent_tags, reduction='sum')
    sn_d2_l = F.binary_cross_entropy(bs_emits_, tags_2, reduction='sum')
    return sn_d1_l, sn_d2_l


def init_the_logger(hdlr):
    if not os.path.exists(odir):
        os.makedirs(odir)
    od = odir.split('/')[-1]  # 'sent_posit_drmm_MarginRankingLoss_0p001'
    logger = logging.getLogger(od)
    if (hdlr is not None):
        logger.removeHandler(hdlr)
    hdlr = logging.FileHandler(os.path.join(odir, 'model.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr


def get_words(s, idf, max_idf):
    sl = tokenize(s)
    sl = [s for s in sl]
    sl2 = [s for s in sl if idf_val(s, idf, max_idf) >= 2.0]
    return sl, sl2


def tokenize(x):
    # return bioclean(x)
    x_tokens = bert_tokenizer.tokenize(x)
    x_tokens = fix_bert_tokens(x_tokens)
    return x_tokens


def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf


def get_embeds(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if (tok in wv):
            ret1.append(tok)
            ret2.append(wv[tok])
    return ret1, np.array(ret2, 'float64')


def get_embeds_use_unk(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if (tok in wv):
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
                qwords_in_doc += 1
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
    qd1 = query_doc_overlap(qwords, dwords, idf, max_idf)
    bm25 = [bm25]
    return qd1[0:3] + bm25


def GetWords(data, doc_text, words):
    for i in tqdm(range(len(data['queries'])), ascii=True):
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
    gold_snips = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            gold_snips.extend(sent_tokenize(sn['text']))
    return list(set(gold_snips))


def prep_extracted_snippets(extracted_snippets, docs, qid, top10docs, quest_body):
    ret = {
        'body': quest_body,
        'documents': top10docs,
        'id': qid,
        'snippets': [],
    }
    for esnip in extracted_snippets:
        pid = esnip[2].split('/')[-1]
        the_text = esnip[3]
        esnip_res = {
            # 'score'     : esnip[1],
            "document": "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pid),
            "text": the_text
        }
        try:
            ind_from = docs[pid]['title'].index(the_text)
            ind_to = ind_from + len(the_text)
            esnip_res["beginSection"] = "title"
            esnip_res["endSection"] = "title"
            esnip_res["offsetInBeginSection"] = ind_from
            esnip_res["offsetInEndSection"] = ind_to
        except:
            # print(the_text)
            # pprint(docs[pid])
            ind_from = docs[pid]['abstractText'].index(the_text)
            ind_to = ind_from + len(the_text)
            esnip_res["beginSection"] = "abstract"
            esnip_res["endSection"] = "abstract"
            esnip_res["offsetInBeginSection"] = ind_from
            esnip_res["offsetInEndSection"] = ind_to
        ret['snippets'].append(esnip_res)
    return ret


def get_snips(quest_id, gid, bioasq6_data):
    good_snips = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            if (sn['document'].endswith(gid)):
                good_snips.extend(sent_tokenize(sn['text']))
    return good_snips


def get_the_mesh(the_doc):
    good_meshes = []
    if ('meshHeadingsList' in the_doc):
        for t in the_doc['meshHeadingsList']:
            t = t.split(':', 1)
            t = t[1].strip()
            t = t.lower()
            good_meshes.append(t)
    elif ('MeshHeadings' in the_doc):
        for mesh_head_set in the_doc['MeshHeadings']:
            for item in mesh_head_set:
                good_meshes.append(item['text'].strip().lower())
    if ('Chemicals' in the_doc):
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
    return int(
        any(
            [
                (one_sent.encode('ascii', 'ignore') in gold_snip.encode('ascii', 'ignore'))
                or
                (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii', 'ignore'))
                for gold_snip in gold_snips
            ]
        )
    )


def create_one_hot_and_sim(tokens1, tokens2):
    '''
    :param tokens1:
    :param tokens2:
    :return:
    exxample call : create_one_hot_and_sim('c d e'.split(), 'a b c'.split())
    '''
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    #
    values = list(set(tokens1 + tokens2))
    integer_encoded = label_encoder.fit_transform(values)
    #
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)
    #
    lab1 = label_encoder.transform(tokens1)
    lab1 = np.expand_dims(lab1, axis=1)
    oh1 = onehot_encoder.transform(lab1)
    #
    lab2 = label_encoder.transform(tokens2)
    lab2 = np.expand_dims(lab2, axis=1)
    oh2 = onehot_encoder.transform(lab2)
    #
    ret = np.matmul(oh1, np.transpose(oh2), out=None)
    #
    return oh1, oh2, ret


def prep_data(quest, the_doc, pid, the_bm25, idf, max_idf):
    good_sents = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
    ####
    bert_f = os.path.join(bert_embeds_dir, '{}.p'.format(pid))
    gemb = pickle.load(open(bert_f, 'rb'))
    # title_bert_average_embeds
    # abs_bert_average_embeds
    # mesh_bert_average_embeds
    ####
    quest_toks = tokenize(quest)
    good_doc_af = GetScores(quest, the_doc['title'] + the_doc['abstractText'], the_bm25, idf, max_idf)
    good_doc_af.append(len(good_sents) / 60.)
    #
    all_bert_embeds = gemb['title_bert_average_embeds'] + gemb['abs_bert_average_embeds']
    doc_embeds = np.concatenate(all_bert_embeds, axis=0)
    doc_toks = []
    for good_text, bert_embeds in zip(good_sents, all_bert_embeds):
        sent_toks = tokenize(good_text)
        doc_toks.extend(sent_toks)
    #
    tomi = (set(doc_toks) & set(quest_toks))
    tomi_no_stop = tomi - set(stopwords)
    BM25score = similarity_score(quest_toks, doc_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
    tomi_no_stop_idfs = [idf_val(w, idf, max_idf) for w in tomi_no_stop]
    tomi_idfs = [idf_val(w, idf, max_idf) for w in tomi]
    quest_idfs = [idf_val(w, idf, max_idf) for w in quest_toks]
    features = [
        len(quest) / 300.,
        len(the_doc['title'] + the_doc['abstractText']) / 300.,
        len(tomi_no_stop) / 100.,
        BM25score,
        sum(tomi_no_stop_idfs) / 100.,
        sum(tomi_idfs) / sum(quest_idfs),
    ]
    good_doc_af.extend(features)
    ####
    oh1, oh2, oh_sim = create_one_hot_and_sim(quest_toks, doc_toks)
    return {
        'doc_af': good_doc_af,
        'doc_embeds': doc_embeds,
        'doc_oh_sim': oh_sim,
        'held_out_sents': good_sents
    }

def train_data_step1(train_data):
    ret = []
    for dato in tqdm(train_data['queries'], ascii=True):
        quest = dato['query_text']
        quest_id = dato['query_id']
        bm25s = {t['doc_id']: t['norm_bm25_score'] for t in dato[u'retrieved_documents']}
        ret_pmids = [t[u'doc_id'] for t in dato[u'retrieved_documents']]
        good_pmids = [t for t in ret_pmids if t in dato[u'relevant_documents']]
        bad_pmids = [t for t in ret_pmids if t not in dato[u'relevant_documents']]
        if (len(bad_pmids) > 0):
            for gid in good_pmids:
                bid = random.choice(bad_pmids)
                ret.append((quest, quest_id, gid, bid, bm25s[gid], bm25s[bid]))
    print('')
    return ret


def fix_bert_tokens(tokens):
    ret = []
    for t in tokens:
        if (t.startswith('##')):
            ret[-1] = ret[-1] + t[2:]
        else:
            ret.append(t)
    return ret


def train_data_step2(instances, docs, bioasq6_data, idf, max_idf, use_sent_tokenizer):
    for quest_text, quest_id, gid, bid, bm25s_gid, bm25s_bid in instances:
        # print(quest_text)
        qemb = all_quest_embeds[quest_text]
        qemb = np.concatenate(qemb, axis=0)
        # pprint(qemb.keys())
        if (use_sent_tokenizer):
            good_snips = get_snips(quest_id, gid, bioasq6_data)
            good_snips = [' '.join(bioclean(sn)) for sn in good_snips]
        else:
            good_snips = []
        #
        quest_text = ' '.join(bioclean(quest_text.replace('\ufeff', ' ')))
        #
        datum = prep_data(quest_text, docs[gid], gid, bm25s_gid, idf, max_idf)
        good_doc_af = datum['doc_af']
        good_doc_embeds = datum['doc_embeds']
        good_oh_sims = datum['doc_oh_sim']
        #
        datum = prep_data(quest_text, docs[bid], bid, bm25s_bid, idf, max_idf)
        bad_doc_af = datum['doc_af']
        bad_doc_embeds = datum['doc_embeds']
        bad_oh_sims = datum['doc_oh_sim']
        #
        quest_tokens = tokenize(quest_text)
        q_idfs = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
        #
        yield {
            'good_doc_af': good_doc_af,
            'good_doc_embeds': good_doc_embeds,
            'good_oh_sims': good_oh_sims,
            #
            'bad_doc_af': bad_doc_af,
            'bad_doc_embeds': bad_doc_embeds,
            'bad_oh_sims': bad_oh_sims,
            #
            'quest_embeds': qemb,
            'q_idfs': q_idfs,
        }


def train_one(epoch, bioasq6_data, two_losses, use_sent_tokenizer):
    model.train()
    batch_costs, batch_acc, epoch_costs, epoch_acc = [], [], [], []
    batch_counter, epoch_aver_cost, epoch_aver_acc = 0, 0., 0.
    #
    train_instances = train_data_step1(train_data)
    random.shuffle(train_instances)
    #
    start_time = time.time()
    pbar = tqdm(
        iterable=train_data_step2(train_instances, train_docs, bioasq6_data, idf, max_idf, use_sent_tokenizer),
        # total=378,
        total=12114,
        ascii=True
    )
    #
    for datum in pbar:
        cost_, doc1_emit_, doc2_emit_ = model(
            good_doc_embeds=datum['good_doc_embeds'],
            bad_doc_embeds=datum['bad_doc_embeds'],
            good_oh_sims=datum['good_oh_sims'],
            bad_oh_sims=datum['bad_oh_sims'],
            question_embeds=datum['quest_embeds'],
            q_idfs=datum['q_idfs'],
            doc_gaf=datum['good_doc_af'],
            doc_baf=datum['bad_doc_af']
        )
        #
        batch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_costs.append(cost_.cpu().item())
        batch_costs.append(cost_)
        if (len(batch_costs) == b_size):
            batch_counter += 1
            batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs,
                                                                                         batch_acc, epoch_acc)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                     batch_aver_acc, epoch_aver_acc, elapsed_time))
            logger.info(
                '{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                   batch_aver_acc, epoch_aver_acc, elapsed_time))
            batch_costs, batch_acc = [], []
    if (len(batch_costs) > 0):
        batch_counter += 1
        batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs,
                                                                                     batch_acc, epoch_acc)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                 batch_aver_acc, epoch_aver_acc, elapsed_time))
        logger.info('{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                       batch_aver_acc, epoch_aver_acc, elapsed_time))
    print('Epoch:{:02d} aver_epoch_cost: {:.4f} aver_epoch_acc: {:.4f}'.format(epoch, epoch_aver_cost, epoch_aver_acc))
    logger.info(
        'Epoch:{:02d} aver_epoch_cost: {:.4f} aver_epoch_acc: {:.4f}'.format(epoch, epoch_aver_cost, epoch_aver_acc))


def do_for_one_retrieved(doc_emit_, gs_emits_, held_out_sents, retr, doc_res, gold_snips):
    emition = doc_emit_.cpu().item()
    if (type(gs_emits_) is list):
        emitss = gs_emits_
    else:
        emitss = gs_emits_.tolist()
    mmax = max(emitss)
    all_emits, extracted_from_one = [], []
    for ind in range(len(emitss)):
        t = (
            snip_is_relevant(held_out_sents[ind], gold_snips),
            emitss[ind],
            "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(retr['doc_id']),
            held_out_sents[ind]
        )
        all_emits.append(t)
        if (emitss[ind] == mmax):
            extracted_from_one.append(t)
    doc_res[retr['doc_id']] = float(emition)
    all_emits = sorted(all_emits, key=lambda x: x[1], reverse=True)
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
    ret = {}
    for es in extracted_snippets:
        if (es[2] in ret):
            if (es[1] > ret[es[2]][1]):
                ret[es[2]] = es
        else:
            ret[es[2]] = es
    sorted_snips = sorted(ret.values(), key=lambda x: x[1], reverse=True)
    return sorted_snips[:10]


def select_snippets_v3(extracted_snippets, the_doc_scores):
    '''
    :param      extracted_snippets:
    :param      doc_res:
    :return:    returns the top 10 snippets across all documents (0..n from each doc)
    '''
    norm_doc_scores = get_norm_doc_scores(the_doc_scores)
    # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
    extracted_snippets = [tt for tt in extracted_snippets if (tt[2] in norm_doc_scores)]
    sorted_snips = sorted(extracted_snippets, key=lambda x: x[1] * norm_doc_scores[x[2]], reverse=True)
    return sorted_snips[:10]


def do_for_some_retrieved(docs, dato, retr_docs, data_for_revision, ret_data, use_sent_tokenizer):
    emitions = {
        'body': dato['query_text'],
        'id': dato['query_id'],
        'documents': []
    }
    #
    quest_text = dato['query_text']
    #
    qemb = all_quest_embeds[quest_text]
    qemb = np.concatenate(qemb, axis=0)
    quest_text = ' '.join(bioclean(quest_text.replace('\ufeff', ' ')))
    #
    quest_tokens = tokenize(quest_text)
    q_idfs = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
    gold_snips = get_gold_snips(dato['query_id'], bioasq6_data)
    #
    doc_res, extracted_snippets = {}, []
    extracted_snippets_known_rel_num = []
    for retr in retr_docs:
        datum = prep_data(quest_text, docs[retr['doc_id']], retr['doc_id'], retr['norm_bm25_score'], idf, max_idf)
        doc_emit_ = model.emit_one(
            good_doc_embeds=datum['doc_embeds'],
            good_oh_sims=datum['doc_oh_sim'],
            question_embeds=qemb,
            q_idfs=q_idfs,
            doc_gaf=datum['doc_af']
        )
        gs_emits_ = [0] * len(datum['held_out_sents'])
        doc_res, extracted_from_one, all_emits = do_for_one_retrieved(
            doc_emit_, gs_emits_, datum['held_out_sents'], retr, doc_res, gold_snips
        )
        # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
        extracted_snippets.extend(extracted_from_one)
        #
        total_relevant = sum([1 for em in all_emits if (em[0] == True)])
        if (total_relevant > 0):
            extracted_snippets_known_rel_num.extend(all_emits[:total_relevant])
        if (dato['query_id'] not in data_for_revision):
            data_for_revision[dato['query_id']] = {'query_text': dato['query_text'],
                                                   'snippets': {retr['doc_id']: all_emits}}
        else:
            data_for_revision[dato['query_id']]['snippets'][retr['doc_id']] = all_emits
    #
    doc_res = sorted(doc_res.items(), key=lambda x: x[1], reverse=True)
    the_doc_scores = dict([("http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]), pm[1]) for pm in doc_res[:10]])
    doc_res = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
    emitions['documents'] = doc_res[:100]
    ret_data['questions'].append(emitions)
    #
    extracted_snippets = [tt for tt in extracted_snippets if (tt[2] in doc_res[:10])]
    extracted_snippets_known_rel_num = [tt for tt in extracted_snippets_known_rel_num if (tt[2] in doc_res[:10])]
    if (use_sent_tokenizer):
        extracted_snippets_v1 = select_snippets_v1(extracted_snippets)
        extracted_snippets_v2 = select_snippets_v2(extracted_snippets)
        extracted_snippets_v3 = select_snippets_v3(extracted_snippets, the_doc_scores)
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
    snips_res_v1 = prep_extracted_snippets(extracted_snippets_v1, docs, dato['query_id'], doc_res[:10],
                                           dato['query_text'])
    snips_res_v2 = prep_extracted_snippets(extracted_snippets_v2, docs, dato['query_id'], doc_res[:10],
                                           dato['query_text'])
    snips_res_v3 = prep_extracted_snippets(extracted_snippets_v3, docs, dato['query_id'], doc_res[:10],
                                           dato['query_text'])
    # pprint(snips_res_v1)
    # pprint(snips_res_v2)
    # pprint(snips_res_v3)
    # exit()
    #
    snips_res_known_rel_num_v1 = prep_extracted_snippets(extracted_snippets_known_rel_num_v1, docs, dato['query_id'],
                                                         doc_res[:10], dato['query_text'])
    snips_res_known_rel_num_v2 = prep_extracted_snippets(extracted_snippets_known_rel_num_v2, docs, dato['query_id'],
                                                         doc_res[:10], dato['query_text'])
    snips_res_known_rel_num_v3 = prep_extracted_snippets(extracted_snippets_known_rel_num_v3, docs, dato['query_id'],
                                                         doc_res[:10], dato['query_text'])
    #
    snips_res = {
        'v1': snips_res_v1,
        'v2': snips_res_v2,
        'v3': snips_res_v3,
    }
    snips_res_known = {
        'v1': snips_res_known_rel_num_v1,
        'v2': snips_res_known_rel_num_v2,
        'v3': snips_res_known_rel_num_v3,
    }
    return data_for_revision, ret_data, snips_res, snips_res_known


def print_the_results(prefix, all_bioasq_gold_data, all_bioasq_subm_data, all_bioasq_subm_data_known,
                      data_for_revision):
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
    ret_data = {'questions': []}
    all_bioasq_subm_data_v1 = {"questions": []}
    all_bioasq_subm_data_known_v1 = {"questions": []}
    all_bioasq_subm_data_v2 = {"questions": []}
    all_bioasq_subm_data_known_v2 = {"questions": []}
    all_bioasq_subm_data_v3 = {"questions": []}
    all_bioasq_subm_data_known_v3 = {"questions": []}
    all_bioasq_gold_data = {'questions': []}
    data_for_revision = {}
    #
    for dato in tqdm(data['queries'], ascii=True):
        all_bioasq_gold_data['questions'].append(bioasq6_data[dato['query_id']])
        data_for_revision, ret_data, snips_res, snips_res_known = do_for_some_retrieved(docs, dato,
                                                                                        dato['retrieved_documents'],
                                                                                        data_for_revision, ret_data,
                                                                                        use_sent_tokenizer)
        all_bioasq_subm_data_v1['questions'].append(snips_res['v1'])
        all_bioasq_subm_data_v2['questions'].append(snips_res['v2'])
        all_bioasq_subm_data_v3['questions'].append(snips_res['v3'])
        all_bioasq_subm_data_known_v1['questions'].append(snips_res_known['v1'])
        all_bioasq_subm_data_known_v2['questions'].append(snips_res_known['v3'])
        all_bioasq_subm_data_known_v3['questions'].append(snips_res_known['v3'])
    #
    print_the_results('v1 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v1, all_bioasq_subm_data_known_v1,
                      data_for_revision)
    print_the_results('v2 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v2, all_bioasq_subm_data_known_v2,
                      data_for_revision)
    print_the_results('v3 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v3, all_bioasq_subm_data_known_v3,
                      data_for_revision)
    #
    if (prefix == 'dev'):
        with open(os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(dataloc + 'bioasq.dev.json',
                              os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json'))
    else:
        with open(os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_test.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(dataloc + 'bioasq.test.json',
                              os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_test.json'))
    return res_map


def load_all_data(dataloc, idf_pickle_path):
    print('loading pickle data')
    #
    with open(dataloc + 'BioASQ-trainingDataset6b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict((q['id'], q) for q in bioasq6_data['questions'])
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
    train_data = RemoveBadYears(train_data, train_docs, True)
    train_data = RemoveTrainLargeYears(train_data, train_docs)
    dev_data = RemoveBadYears(dev_data, dev_docs, False)
    test_data = RemoveBadYears(test_data, test_docs, False)
    #
    if (os.path.exists(bert_all_words_path)):
        words = pickle.load(open(bert_all_words_path, 'rb'))
    else:
        words = {}
        GetWords(train_data, train_docs, words)
        GetWords(dev_data, dev_docs, words)
        GetWords(test_data, test_docs, words)
        pickle.dump(words, open(bert_all_words_path, 'wb'), protocol=2)
    #
    print('loading idfs')
    idf, max_idf = load_idfs(idf_pickle_path, words)
    return test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, bioasq6_data


class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim=30, k_for_maxpool=5):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k = k_for_maxpool
        #
        self.doc_add_feats = 11
        self.embedding_dim = embedding_dim
        # to create q weights
        self.init_context_module()
        self.init_question_weight_module()
        self.init_mlps_for_pooled_attention()
        self.init_doc_out_layer()
        # doc loss func
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def init_context_module(self):
        self.trigram_conv_1 = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.trigram_conv_2 = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_2 = torch.nn.LeakyReLU(negative_slope=0.1)

    def init_question_weight_module(self):
        self.q_weights_mlp = nn.Linear(self.embedding_dim + 1, 1, bias=True)

    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1 = nn.Linear(3 * 3, 8, bias=True)
        self.my_relu1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2 = nn.Linear(8, 1, bias=True)

    def init_doc_out_layer(self):
        self.final_layer = nn.Linear(self.doc_add_feats + 1, 1, bias=True)

    def init_sent_output_layer(self):
        if (self.context_method == 'MLP'):
            self.sent_out_layer = nn.Linear(4, 1, bias=False)
        else:
            self.sent_res_h0 = autograd.Variable(torch.randn(2, 1, 5))
            self.sent_res_bigru = nn.GRU(input_size=4, hidden_size=5, bidirectional=True, batch_first=False)
            self.sent_res_mlp = nn.Linear(10, 1, bias=False)

    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos

    def apply_context_convolution(self, the_input, the_filters, activation):
        conv_res = the_filters(the_input.transpose(0, 1).unsqueeze(0))
        if (activation is not None):
            conv_res = activation(conv_res)
        pad = the_filters.padding[0]
        ind_from = int(np.floor(pad / 2.0))
        ind_to = ind_from + the_input.size(0)
        conv_res = conv_res[:, :, ind_from:ind_to]
        conv_res = conv_res.transpose(1, 2)
        conv_res = conv_res + the_input
        return conv_res.squeeze(0)

    def my_cosine_sim(self, A, B):
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        A_mag = torch.norm(A, 2, dim=2)
        B_mag = torch.norm(B, 2, dim=2)
        num = torch.bmm(A, B.transpose(-1, -2))
        den = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
        dist_mat = num / den
        return dist_mat

    def pooling_method(self, sim_matrix):
        sorted_res = torch.sort(sim_matrix, -1)[0]  # sort the input minimum to maximum
        k_max_pooled = sorted_res[:, -self.k:]  # select the last k of each instance in our data
        average_k_max_pooled = k_max_pooled.sum(-1) / float(self.k)  # average these k values
        the_maximum = k_max_pooled[:, -1]  # select the maximum value of each instance
        the_average_over_all = sorted_res.sum(-1) / float(
            sim_matrix.size(1))  # add average of all elements as long sentences might have more matches
        the_concatenation = torch.stack([the_maximum, average_k_max_pooled, the_average_over_all],
                                        dim=-1)  # concatenate maximum value and average of k-max values
        return the_concatenation  # return the concatenation

    def get_output(self, input_list, weights):
        temp = torch.cat(input_list, -1)
        lo = self.linear_per_q1(temp)
        lo = self.my_relu1(lo)
        lo = self.linear_per_q2(lo)
        lo = lo.squeeze(-1)
        lo = lo * weights
        sr = lo.sum(-1) / lo.size(-1)
        return sr

    def apply_sent_res_bigru(self, the_input):
        output, hn = self.sent_res_bigru(the_input.unsqueeze(1), self.sent_res_h0)
        output = self.sent_res_mlp(output)
        return output.squeeze(-1).squeeze(-1)

    def get_max_and_average_of_k_max(self, res, k):
        sorted_res = torch.sort(res)[0]
        k_max_pooled = sorted_res[-k:]
        average_k_max_pooled = k_max_pooled.sum() / float(k)
        the_maximum = k_max_pooled[-1]
        # print(the_maximum)
        # print(the_maximum.size())
        # print(average_k_max_pooled)
        # print(average_k_max_pooled.size())
        the_concatenation = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation

    def emit_doc_cnn(self, doc_embeds, question_embeds, q_conv_res_trigram, q_weights, sim_oh):
        conv_res = self.apply_context_convolution(doc_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        conv_res = self.apply_context_convolution(conv_res, self.trigram_conv_2, self.trigram_conv_activation_2)
        sim_insens = self.my_cosine_sim(question_embeds, doc_embeds).squeeze(0)
        sim_sens = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
        insensitive_pooled = self.pooling_method(sim_insens)
        sensitive_pooled = self.pooling_method(sim_sens)
        oh_pooled = self.pooling_method(sim_oh)
        doc_emit = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
        doc_emit = doc_emit.unsqueeze(-1)
        return doc_emit

    def emit_doc_bigru(self, doc_embeds, question_embeds, q_conv_res_trigram, q_weights):
        conv_res, hn = self.apply_context_gru(doc_embeds, self.context_h0)
        sim_insens = self.my_cosine_sim(question_embeds, doc_embeds).squeeze(0)
        sim_oh = (sim_insens > (1 - (1e-3))).float()
        sim_sens = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
        insensitive_pooled = self.pooling_method(sim_insens)
        sensitive_pooled = self.pooling_method(sim_sens)
        oh_pooled = self.pooling_method(sim_oh)
        doc_emit = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
        doc_emit = doc_emit.unsqueeze(-1)
        return doc_emit

    def get_max(self, res):
        return torch.max(res)

    def get_kmax(self, res):
        res = torch.sort(res, 0)[0]
        res = res[-self.k2:].squeeze(-1)
        if (res.size()[0] < self.k2):
            res = torch.cat([res, torch.zeros(self.k2 - res.size()[0])], -1)
        return res

    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res

    def get_maxmin_max(self, res):
        res = self.min_max_norm(res)
        res = torch.max(res)
        return res

    def emit_one(self, good_doc_embeds, good_oh_sims, question_embeds, q_idfs, doc_gaf):
        good_oh_sims = autograd.Variable(torch.FloatTensor(good_oh_sims), requires_grad=False)
        q_idfs = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False)
        question_embeds = autograd.Variable(torch.FloatTensor(question_embeds), requires_grad=False)
        doc_gaf = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False)
        good_doc_embeds = autograd.Variable(torch.FloatTensor(good_doc_embeds), requires_grad=False)
        # HANDLE QUESTION
        q_context = self.apply_context_convolution(question_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context = self.apply_context_convolution(q_context, self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights = torch.cat([q_context, q_idfs], -1)
        q_weights = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights = F.softmax(q_weights, dim=-1)
        # HANDLE DOCS
        good_out = self.emit_doc_cnn(good_doc_embeds, question_embeds, q_context, q_weights, good_oh_sims)
        #
        good_out_pp = torch.cat([good_out, doc_gaf], -1)
        #
        final_good_output = self.final_layer(good_out_pp)
        return final_good_output

    def forward(
            self, good_doc_embeds, bad_doc_embeds, good_oh_sims, bad_oh_sims, question_embeds, q_idfs, doc_gaf, doc_baf
    ):
        good_oh_sims = autograd.Variable(torch.FloatTensor(good_oh_sims), requires_grad=False)
        bad_oh_sims = autograd.Variable(torch.FloatTensor(bad_oh_sims), requires_grad=False)
        q_idfs = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False)
        question_embeds = autograd.Variable(torch.FloatTensor(question_embeds), requires_grad=False)
        doc_gaf = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False)
        doc_baf = autograd.Variable(torch.FloatTensor(doc_baf), requires_grad=False)
        good_doc_embeds = autograd.Variable(torch.FloatTensor(good_doc_embeds), requires_grad=False)
        bad_doc_embeds = autograd.Variable(torch.FloatTensor(bad_doc_embeds), requires_grad=False)
        # HANDLE QUESTION
        q_context = self.apply_context_convolution(question_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context = self.apply_context_convolution(q_context, self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights = torch.cat([q_context, q_idfs], -1)
        q_weights = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights = F.softmax(q_weights, dim=-1)
        # HANDLE DOCS
        good_out = self.emit_doc_cnn(good_doc_embeds, question_embeds, q_context, q_weights, good_oh_sims)
        bad_out = self.emit_doc_cnn(bad_doc_embeds, question_embeds, q_context, q_weights, bad_oh_sims)
        #
        good_out_pp = torch.cat([good_out, doc_gaf], -1)
        bad_out_pp = torch.cat([bad_out, doc_baf], -1)
        #
        final_good_output = self.final_layer(good_out_pp)
        final_bad_output = self.final_layer(bad_out_pp)
        #
        loss1 = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output


use_cuda = torch.cuda.is_available()

# atlas , cslab243
bert_all_words_path = '/home/dpappas/bioasq_all/bert_all_words.pkl'
all_quest_embeds = pickle.load(open('/home/dpappas/bioasq_all/all_quest_bert_embeds_after_pca.p', 'rb'))
idf_pickle_path = '/home/dpappas/bioasq_all/idf.pkl'
dataloc = '/home/dpappas/bioasq_all/bioasq_data/'
bert_embeds_dir = '/home/dpappas/bioasq_all/bert_embeds_after_pca/'
eval_path = '/home/dpappas/bioasq_all/eval/run_eval.py'
retrieval_jar_path = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'
odd = '/home/dpappas/'
use_cuda = False
bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

k_for_maxpool = 5
k_sent_maxpool = 5
embedding_dim = 50  # 30  # 200
lr = 0.01
b_size = 32
max_epoch = 10

hdlr = None
for run in range(0, 5):
    #
    my_seed = run
    random.seed(my_seed)
    torch.manual_seed(my_seed)
    #
    odir = 'sigir_dec_ret_pdrmm_bert_2L_no_mesh_0p01_run_{}/'.format(run)
    odir = os.path.join(odd, odir)
    print(odir)
    if (not os.path.exists(odir)):
        os.makedirs(odir)
    #
    logger, hdlr = init_the_logger(hdlr)
    print('random seed: {}'.format(my_seed))
    logger.info('random seed: {}'.format(my_seed))
    #
    (
        test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, bioasq6_data
    ) = load_all_data(dataloc=dataloc, idf_pickle_path=idf_pickle_path)
    #
    avgdl, mean, deviation = get_bm25_metrics(avgdl=21.2508, mean=0.5973, deviation=0.5926)
    print(avgdl, mean, deviation)
    #
    print('Compiling model...')
    logger.info('Compiling model...')
    model = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool)
    if (use_cuda):
        model = model.cuda()
    params = model.parameters()
    print_params(model)
    optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #
    best_dev_map, test_map = None, None
    for epoch in range(max_epoch):
        train_one(epoch + 1, bioasq6_data, two_losses=True, use_sent_tokenizer=True)
        epoch_dev_map = get_one_map('dev', dev_data, dev_docs, use_sent_tokenizer=True)
        if (best_dev_map is None or epoch_dev_map >= best_dev_map):
            best_dev_map = epoch_dev_map
            test_map = get_one_map('test', test_data, test_docs, use_sent_tokenizer=True)
            save_checkpoint(epoch, model, best_dev_map, optimizer,
                            filename=os.path.join(odir, 'best_checkpoint.pth.tar'))
        print('epoch:{:02d} epoch_dev_map:{:.4f} best_dev_map:{:.4f} test_map:{:.4f}'.format(epoch + 1, epoch_dev_map,
                                                                                             best_dev_map, test_map))
        logger.info(
            'epoch:{:02d} epoch_dev_map:{:.4f} best_dev_map:{:.4f} test_map:{:.4f}'.format(epoch + 1, epoch_dev_map,
                                                                                           best_dev_map, test_map))
