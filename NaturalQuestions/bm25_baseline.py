
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

dev_data, dev_docs, test_data, test_docs, train_data, train_docs, bioasq7_data = load_all_data(dataloc)

f_out               = '/home/dpappas/NQ_data/bm25_results.json'

###########################################################

bm25_data   = {'questions': []}
for q in test_data['queries']:
    ###############################
    q_text      = q['query_text']
    docs        = [d['doc_id'] for d in q['retrieved_documents']]
    all_sents       = []
    all_text_sents  = []
    for did in docs:
        for sent in sent_tokenize(test_docs[did]['title']):
            all_sents.append(
                (
                    "title",
                    "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(did),
                    "title",
                    test_docs[did]['title'].index(sent),
                    test_docs[did]['title'].index(sent)+len(sent),
                    sent
                )
            )
            all_text_sents.append(sent)
        for sent in sent_tokenize(test_docs[did]['abstractText']):
            all_sents.append(
                (
                    "abstract",
                    "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(did),
                    "abstract",
                    test_docs[did]['abstractText'].index(sent),
                    test_docs[did]['abstractText'].index(sent)+len(sent),
                    sent
                )
            )
            all_text_sents.append(sent)
    ###############################
    tokenized_corpus    = [doc.lower().split() for doc in all_text_sents]
    bm25                = BM25Okapi(tokenized_corpus)
    doc_scores          = bm25.get_scores(q_text.lower().split())
    max_inds            = (np.array(doc_scores)).argsort()[-10:][::-1].tolist()
    retr_sents          = [all_sents[index] for index in max_inds]
    ###############################
    retr_snips          = [
        {
            "beginSection"          : snipi[0],
            "document"              : snipi[1],
            "endSection"            : snipi[2],
            "offsetInBeginSection"  : snipi[3],
            "offsetInEndSection"    : snipi[4],
            "text"                  : snipi[5],
        }
        for snipi in retr_sents
    ]
    bm25_data['questions'].append(
        {
            "body"      : "n/a",
            "id"        : q['query_id'],
            "documents" : ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(doc_id) for doc_id in docs[:10]],
            "snippets"  : retr_snips
        }
    )

with open(f_out, 'w') as f:
    f.write(json.dumps(bm25_data, indent=4, sort_keys=False))
    f.close()

'''

'''


