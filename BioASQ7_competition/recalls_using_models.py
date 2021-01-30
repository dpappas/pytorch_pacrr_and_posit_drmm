
import pickle, json
import numpy as np
import re

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())

def snip_is_relevant(one_sent, gold_snips):
    # return int(
    #     any(
    #         [
    #             fuzz.ratio(one_sent, gold_snip) > 0.95 for gold_snip in gold_snips
    #         ]
    #     )
    # )
    return int(
        any(
            [
                # (one_sent.encode('ascii', 'ignore')  in gold_snip.encode('ascii','ignore'))
                # or
                # (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii','ignore'))
                (one_sent in gold_snip or gold_snip in one_sent)
                and
                (
                        len(one_sent) < len(gold_snip)+6 and
                        len(one_sent) > len(gold_snip) - 6
                )
                for gold_snip in gold_snips
            ]
        )
    )

for batch_no in range(1,6):
    d = json.load(open('/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_{}/pdrmm_bcnn.json'.format(batch_no)))
    # d 	= json.load(open('/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_{}/bert_jpdrmm.json'.format(batch_no,batch_no)))
    # d = json.load(
    #     open(
    #         '/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_adapt_run_frozen/batch_{}/v3 test_emit_bioasq.json'.format(batch_no)
    #     )
    # )
    # d = json.load(open('/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_toponly_unfrozen_run_0/batch_{}/v3 test_emit_bioasq.json'.format(batch_no)))
    d2 = json.load(open('bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(batch_no, batch_no)))
    id2rel      = {}
    id2relsnip  = {}
    for q in d2['questions']:
        id2rel[q['id']]     = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in q['documents']]
        id2relsnip[q['id']] = [bioclean(snip['text'].strip()) for snip in q['snippets']]
    ################################################################################################################################
    recalls         = []
    recalls_snip    = []
    for quer in d['questions']:
        try:
            quer['documents'] = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in quer['documents']]
            retr_ids    = [doc for doc in quer['documents'][:10] if doc in id2rel[quer['id']] ]
        except:
            retr_ids    = []
        #################
        found           = float(len(retr_ids))
        tot             = float(len(id2rel[quer['id']]))
        recalls.append(found/tot)
        #################
        retr_snips      = [bioclean(snip['text']) for snip in quer['snippets'][:10]]
        retr_snips      = [t for t in retr_snips if snip_is_relevant(t, id2relsnip[quer['id']])]
        if(len(id2relsnip[quer['id']]) != 0):
            recalls_snip.append(float(len(retr_snips))/float(len(id2relsnip[quer['id']])))
        #################
    print(np.average(recalls))
    print(np.average(recalls_snip))
    ################################################################################################################################


