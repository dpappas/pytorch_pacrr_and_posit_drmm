

import  json
from    pprint import pprint

# gold_f = "/home/dpappas/test_bert_jpdrmm/v3 test_gold_bioasq.json"
# emit_f = "/home/dpappas/test_bert_jpdrmm/v3 test_emit_bioasq.json"

gold_f      = "/home/DATA/Biomedical/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1"
emit_f_docs = '/home/DATA/Biomedical/bioasq7/document_results/test_batch_1/bert.json'
emit_f_sent = "/home/DATA/Biomedical/bioasq7/snippet_results/test_batch_1/bert_bcnn.json"

all_gd      = json.load(open(gold_f))
all_gd      = dict((q['id'], q) for q in all_gd['questions'])
#############
all_em_docs = json.load(open(emit_f_docs))
all_em_docs = dict((q['id'], q) for q in all_em_docs['questions'])
#############
all_em_sent = json.load(open(emit_f_sent))
all_em_sent = dict((q['id'], q) for q in all_em_sent['questions'])

for id in all_gd:
    gd      = all_gd[id]
    qtext   = gd['body']
    ed      = all_em_docs[id]
    s1      = set(gd['documents'])
    s2      = set(ed['documents'])
    dif     = s2.difference(s1)
    for doc_link in dif:
        extr_sents = [sn['text'] for sn in all_em_sent[id]['snippets'] if(sn['document'] == doc_link)]
        for extr_sent in extr_sents:
            print(qtext)
            print(doc_link)
            pprint(extr_sent)
            print(20*'-')
    print(20*'=')
