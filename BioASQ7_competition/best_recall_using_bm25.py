
import pickle, json
import numpy as np
import re

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())

def snip_is_relevant(one_sent, gold_snips):
    # print one_sent
    # pprint(gold_snips)
    return int(
        any(
            [
                (one_sent.encode('ascii', 'ignore')  in gold_snip.encode('ascii','ignore'))
                or
                (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii','ignore'))
                for gold_snip in gold_snips
            ]
        )
    )

for batch_no in range(1,6):
    d     = pickle.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'.format(batch_no), 'rb'))
    d2     = json.load(open('bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(batch_no,batch_no)))
    ##################################################################################################
    id2rel      = {}
    id2relsnip  = {}
    for q in d2['questions']:
        id2rel[q['id']]     = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in q['documents']]
        id2relsnip[q['id']] = [bioclean(snip['text'].strip()) for snip in q['snippets']]
    ##################################################################################################
    recalls         = []
    recalls_snip    = []
    # for quer in d['queries']:
    #     retr_ids     = [doc['doc_id'] for doc in quer['retrieved_documents'][:10] if doc['doc_id'] in id2rel[quer['query_id']]]
    #     found         = float(len(retr_ids))
    #     tot         = float(len(id2rel[quer['query_id']]))
    #     recalls.append(found/tot)
    for quer in d['queries']:
        retr_ids    = [doc['doc_id'] for doc in quer['retrieved_documents'][:100] if doc['doc_id'] in id2rel[quer['query_id']]]
        found       = float(len(retr_ids))
        tot         = float(len(id2rel[quer['query_id']]))
        recalls.append(found/tot)
        #################
        retr_snips      = [bioclean(snip['text']) for snip in quer['snippets'][:100]]
        retr_snips      = [t for t in retr_snips if snip_is_relevant(t, id2relsnip[quer['id']])]
        recalls_snip.append(float(len(retr_snips))/float(len(id2relsnip[quer['id']])))
        #################
    ##################################################################################################
    print(np.average(recalls))
    print(np.average(recalls_snip))
