
from rank_bm25 import BM25Okapi
import pickle, json, sys, re
import numpy as np
from pprint import pprint
from    nltk.tokenize               import sent_tokenize
import numpy as np

dataloc = '/home/dpappas/NQ_data/'

def doc_precision_at_k(related_lists, k):
    all_precs = []
    for related_list in related_lists:
        all_precs.append(float(sum(related_list[:k])) / float(k))
    return float(sum(all_precs)) / float(len(all_precs))

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    ###########################################################################################
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    ###########################################################################################
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    ###########################################################################################
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    ###########################################################################################
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def load_all_data(dataloc):
    print('loading pickle data')
    ########################################################
    with open(dataloc+'NQ_training7b.train.dev.test.json', 'r') as f:
        bioasq7_data    = json.load(f)
        bioasq7_data    = dict((q['id'], q) for q in bioasq7_data['questions'])
    ########################################################
    with open(dataloc + 'NQ_bioasq7_bm25_top100.train.pkl', 'rb') as f:
        train_data      = pickle.load(f)
    with open(dataloc + 'NQ_bioasq7_bm25_top100.dev.pkl', 'rb') as f:
        dev_data        = pickle.load(f)
    with open(dataloc + 'NQ_bioasq7_bm25_top100.test.pkl', 'rb') as f:
        test_data       = pickle.load(f)
    ########################################################
    with open(dataloc + 'NQ_bioasq7_bm25_docset_top100.train.dev.test.pkl', 'rb') as f:
        train_docs      = pickle.load(f)
    ########################################################
    dev_data['queries']     = dev_data['queries'][:400] # GIA NA MH MOY PAREI KANA XRONO!
    test_data['queries']    = test_data['queries'][:400] # GIA NA MH MOY PAREI KANA XRONO!
    ########################################################
    dev_docs    = train_docs
    test_docs   = train_docs
    ########################################################
    return dev_data, dev_docs, test_data, test_docs, train_data, train_docs, bioasq7_data

def sent_is_rel(sent, relevant_snips):
    for s in relevant_snips:
        if(s in sent or sent in s):
            return True
    return False

dev_data, dev_docs, test_data, test_docs, train_data, train_docs, bioasq7_data = load_all_data(dataloc)

all_emitted = []
for q in test_data['queries']:
    q_text      = q['query_text']
    qid         = q['query_id']
    docs        = [d['doc_id'] for d in q['retrieved_documents']]
    ###############################
    all_text_sents  = []
    relevant_snips = [ll['text'] for ll in bioasq7_data[qid]['snippets']]
    for did in docs:
        for sent in sent_tokenize(test_docs[did]['title']):
            all_text_sents.append(sent)
        for sent in sent_tokenize(test_docs[did]['abstractText']):
            all_text_sents.append(sent)
    ###############################
    tokenized_corpus    = [doc.lower().split() for doc in all_text_sents]
    bm25                = BM25Okapi(tokenized_corpus)
    doc_scores          = bm25.get_scores(q_text.lower().split())
    max_inds            = (np.array(doc_scores)).argsort()[-10:][::-1].tolist()
    retr_sents          = [all_text_sents[index] for index in max_inds]
    emitted             = [int(sent_is_rel(sent, relevant_snips)) for sent in retr_sents]
    ###############################
    all_emitted.append(emitted)

print(mean_reciprocal_rank(all_emitted))
print(doc_precision_at_k(all_emitted, 1))
print(doc_precision_at_k(all_emitted, 10))
print(np.average([doc_precision_at_k(all_emitted, k) for k in range(1, 11)]))

