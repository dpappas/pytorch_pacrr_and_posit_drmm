
import pickle, json
import numpy as np

for batch_no in range(1,6):
    # d 	= json.load(open('bioasq_all/bioasq7/document_results/test_batch_{}/JBERT.json'.format(batch_no,batch_no)))
    d = json.load(open('/media/dpappas/dpappas_data/models_out/./bioasq7_bertjpdrmadaptnf_toponly_unfrozen_run_0/batch_{}/v3 test_emit_bioasq.json'.format(batch_no)))
    d2 = json.load(open('bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(batch_no, batch_no)))
    id2rel = {}
    for q in d2['questions']:
        id2rel[q['id']] = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in q['documents']]
    ################################################################################################################################
    recalls = []
    for quer in d['questions']:
        quer['documents'] = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in quer['documents']]
        retr_ids    = [
            doc for doc in quer['documents'][:10]
            if doc in id2rel[quer['id']]
        ]
        found 		= float(len(retr_ids))
        tot 		= float(len(id2rel[quer['id']]))
        recalls.append(found/tot)
    print(np.average(recalls))
    ################################################################################################################################


