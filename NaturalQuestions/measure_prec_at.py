
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

def snip_precision_at_k(emited_snippets, answer_spans, k):
    found = 0
    for i in range(k):
        for ans_span in answer_spans:
            if(emited_snippets[i]['document'] == ans_span['document'] and ans_span['text'] in emited_snippets[i]['text']):
                found+=1
    return float(found) / float(k)

def snip_recall_at_k(emited_snippets, answer_spans, k):
    found = 0
    for i in range(k):
        for ans_span in answer_spans:
            if(emited_snippets[i]['document'] == ans_span['document'] and ans_span['text'] in emited_snippets[i]['text']):
                found+=1
    return float(found) / float(len(answer_spans))

ed = json.load(open(emit_fpath))
gd = json.load(open(gold_fpath))

ed = dict((q['id'], q) for q in ed["questions"])
gd = dict((q['id'], q) for q in gd["questions"])

pprint(list(ed.items())[0])
pprint(list(gd.items())[0])

for k in range(1, 11):
    average_doc_pre_at_k    = []
    average_doc_rec_at_k    = []
    average_snip_pre_at_k   = []
    average_snip_rec_at_k   = []
    for id in ed:
        gdocs = gd[id][u'documents']
        edocs = ed[id][u'documents']
        average_doc_pre_at_k.append(doc_precision_at_k(edocs, gdocs, k))
        average_doc_rec_at_k.append(doc_recall_at_k(edocs, gdocs, k))
        average_snip_pre_at_k.append(doc_precision_at_k(edocs, gdocs, k))
        average_snip_rec_at_k.append(doc_recall_at_k(edocs, gdocs, k))
    #######################
    aver_doc_pre        = sum(average_doc_pre_at_k)/ len(average_doc_pre_at_k)
    aver_doc_rec        = sum(average_doc_rec_at_k)/ len(average_doc_rec_at_k)
    aver_snip_pre       = sum(average_snip_pre_at_k)/ len(average_snip_pre_at_k)
    aver_snip_rec       = sum(average_snip_rec_at_k)/ len(average_snip_rec_at_k)
    #######################
    # print(k, aver_doc_pre, aver_doc_rec, aver_snip_pre, aver_snip_rec)
    print(k, aver_doc_pre, aver_doc_rec)
    #######################

# (# of recommended items @k that are relevant) / (# of recommended items @k)





