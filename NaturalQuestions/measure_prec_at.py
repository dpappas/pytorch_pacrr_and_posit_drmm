
import json
from pprint import pprint

emit_fpath = '/home/dpappas/natural_questions_jpdrmm_2L_0p01_run_0/v3 dev_emit_bioasq.json'
gold_fpath = '/home/dpappas/natural_questions_jpdrmm_2L_0p01_run_0/v3 dev_gold_bioasq.json'

def doc_precision_at_k(emited_docs, relevant_docs, k):
    found = 0
    for i in range(k):
        if(emited_docs[i] in relevant_docs):
            found+=1
    return float(found) / float(k)

def doc_recall_at_k(emited_docs, relevant_docs, k):
    found = 0
    for i in range(k):
        if(emited_docs[i] in relevant_docs):
            found+=1
    return float(found) / float(len(relevant_docs))

ed = json.load(open(emit_fpath))
gd = json.load(open(gold_fpath))

ed = dict((q['id'], q) for q in ed["questions"])
gd = dict((q['id'], q) for q in gd["questions"])


for k in range(1, 11):
    average_pre_at_k = []
    average_rec_at_k = []
    for id in ed:
        gdocs = gd[id][u'documents']
        edocs = ed[id][u'documents']
        average_pre_at_k.append(doc_precision_at_k(edocs, gdocs, k))
        average_rec_at_k.append(doc_recall_at_k(edocs, gdocs, k))
    aver_pre = sum(average_pre_at_k)/ len(average_pre_at_k)
    aver_rec = sum(average_rec_at_k)/ len(average_rec_at_k)
    print(k, aver_pre, aver_rec)

# (# of recommended items @k that are relevant) / (# of recommended items @k)





