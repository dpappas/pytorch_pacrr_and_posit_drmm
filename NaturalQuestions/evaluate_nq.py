
import json, pickle
from pprint import pprint
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
    dev_docs    = train_docs
    test_docs   = train_docs
    ########################################################
    return dev_data, dev_docs, test_data, test_docs, train_data, train_docs, bioasq7_data

dev_data, dev_docs, test_data, test_docs, train_data, train_docs, bioasq7_data = load_all_data(dataloc)
test_data   = dict((t['query_id'], t) for t in test_data['queries'])
dev_data    = dict((t['query_id'], t) for t in dev_data['queries'])

# DEV
related_lists = [
    [
        int(tt['is_relevant'])
        for tt in item['retrieved_documents']
    ][:10]
    for item in dev_data.values()
]

print(mean_reciprocal_rank(related_lists))  ### 0.3089
print(doc_precision_at_k(related_lists, 10))  ### 0.0602
print(np.average([doc_precision_at_k(related_lists, k) for k in range(1, 11)])) ### 0.1023
#####################################################################################################
extracted       = json.load(open("/home/dpappas/natural_questions_jpdrmm_2L_0p01_run_0/v3 dev_emit_bioasq.json"))
#####################################################################################################

snip_related_lists = []
for item in extracted['questions']:
    snip_list = []
    for retr_snip in item['snippets']:
        found = False
        for gold_snip in bioasq7_data[item['id']]['snippets']:
            if(retr_snip['document'] == gold_snip['document']):
                if(gold_snip['text'] in retr_snip['text'] or retr_snip['text'] in gold_snip['text']):
                    found = True
                    break
        snip_list.append(int(found))
    snip_related_lists.append(snip_list)

#####################################################################################################

related_lists   = [
    [
        int(t.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '')
        in dev_data[item['id']]['relevant_documents'])
        for t in item['documents']
    ]
    for item in extracted['questions']
]

#####################################################################################################
print(mean_reciprocal_rank(related_lists))  ### 0.4056
print(doc_precision_at_k(related_lists, 10))  ### 0.0710
print(np.average([doc_precision_at_k(related_lists, k) for k in range(1, 11)])) ### 0.1327
#####################################################################################################
print(mean_reciprocal_rank(snip_related_lists))  ### 0.4056
print(doc_precision_at_k(snip_related_lists, 10))  ### 0.0710
print(np.average([doc_precision_at_k(snip_related_lists, k) for k in range(1, 11)])) ### 0.1327
#####################################################################################################

# TEST
related_lists = [
    [
        int(tt['is_relevant'])
        for tt in item['retrieved_documents']
    ][:10]
    for item in test_data.values()
]

print(mean_reciprocal_rank(related_lists)) ### 0.3162
print(doc_precision_at_k(related_lists, 10)) ### 0.0608
print(np.average([doc_precision_at_k(related_lists, k) for k in range(1, 11)])) ### 0.1047
#####################################################################################################
extracted   = json.load(open("/home/dpappas/test_natural_questions_jpdrmm_2L_0p01_run_0/v3 test_emit_bioasq.json"))
# extracted   = json.load(open("/home/dpappas/test_NQ_JBERT/v3 test_emit_bioasq.json"))
#####################################################################################################

snip_related_lists = []
for item in extracted['questions']:
    snip_list = []
    for retr_snip in item['snippets']:
        found = False
        for gold_snip in bioasq7_data[item['id']]['snippets']:
            if(retr_snip['document'] == gold_snip['document']):
                if(gold_snip['text'] in retr_snip['text'] or retr_snip['text'] in gold_snip['text']):
                    found = True
                    break
        snip_list.append(int(found))
    snip_related_lists.append(snip_list)

#####################################################################################################

related_lists = [
    [
        int(t.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '')
        in test_data[item['id']]['relevant_documents'])
        for t in item['documents']
    ]
    for item in extracted['questions']
]

#####################################################################################################
print(mean_reciprocal_rank(related_lists))                                      ### 0.4026
print(doc_precision_at_k(related_lists, 10))                                    ### 0.0703
print(np.average([doc_precision_at_k(related_lists, k) for k in range(1, 11)])) ### 0.1313
#####################################################################################################
print(mean_reciprocal_rank(snip_related_lists))  ### 0.4056
print(doc_precision_at_k(snip_related_lists, 10))  ### 0.0710
print(np.average([doc_precision_at_k(snip_related_lists, k) for k in range(1, 11)])) ### 0.1327
#####################################################################################################


'''
java -Xmx10G -cp \
/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar \
evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/test_natural_questions_jpdrmm_2L_0p01_run_0/v3 test_gold_bioasq.json" \
"/home/dpappas/test_natural_questions_jpdrmm_2L_0p01_run_0/v3 test_emit_bioasq.json"
'''


