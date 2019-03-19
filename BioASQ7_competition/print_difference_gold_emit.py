

import  json
from    pprint import pprint

# gold_f = "/home/dpappas/test_bert_jpdrmm/v3 test_gold_bioasq.json"
# emit_f = "/home/dpappas/test_bert_jpdrmm/v3 test_emit_bioasq.json"

############################################################################################
# gold_f      = "/home/DATA/Biomedical/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1"
# emit_f_docs = '/home/DATA/Biomedical/bioasq7/document_results/test_batch_1/bert.json'
# emit_f_sent = "/home/DATA/Biomedical/bioasq7/snippet_results/test_batch_1/bert_bcnn.json"
# out_path    = "/home/dpappas/bert_bcnn_errors.txt"
############################################################################################
# gold_f      = "/home/DATA/Biomedical/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1"
# emit_f_docs = '/home/DATA/Biomedical/bioasq7/document_results/test_batch_1/term-pacrr.json'
# emit_f_sent = "/home/DATA/Biomedical/bioasq7/snippet_results/test_batch_1/term_pacrr__bcnn.json"
# out_path    = "/home/dpappas/term_pacrr_bcnn_errors.txt"
############################################################################################
gold_f      = "/home/DATA/Biomedical/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1"
emit_f_docs = '/home/DATA/Biomedical/bioasq7/document_results/test_batch_1/bert_jpdrmm.json'
emit_f_sent = '/home/DATA/Biomedical/bioasq7/document_results/test_batch_1/bert_jpdrmm.json'
out_path    = "/home/dpappas/bert_jpdrmm_errors.txt"
############################################################################################
# gold_f      = "/home/DATA/Biomedical/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1"
# emit_f_docs = '/home/DATA/Biomedical/bioasq7/document_results/test_batch_1/jpdrmm.json'
# emit_f_sent = '/home/DATA/Biomedical/bioasq7/document_results/test_batch_1/jpdrmm.json'
# out_path    = "/home/dpappas/jpdrmm_errors.txt"
############################################################################################

#############
all_gd      = json.load(open(gold_f))
all_gd      = dict((q['id'], q) for q in all_gd['questions'])
#############
all_em_docs = json.load(open(emit_f_docs))
all_em_docs = dict((q['id'], q) for q in all_em_docs['questions'])
#############
all_em_sent = json.load(open(emit_f_sent))
all_em_sent = dict((q['id'], q) for q in all_em_sent['questions'])
#############

with open(out_path, 'w') as f:
    for qid in sorted(all_gd.keys()):
        gd      = all_gd[qid]
        qtext   = gd['body']
        ed      = all_em_docs[qid]
        s1      = set(gd['documents'])
        s2      = set(ed['documents'])
        dif     = s2.difference(s1)
        for doc_link in dif:
            extr_sents = [sn['text'] for sn in all_em_sent[qid]['snippets'] if(sn['document'] == doc_link)]
            for extr_sent in extr_sents:
                f.write("{}\n{}\n{}\n{}\n{}\n".format(qid, doc_link, qtext.encode('utf-8'), extr_sent.encode('utf-8'), 20*'-'))
        f.write("{}\n".format(20*'='))
    f.close()
