
import pickle, json
import numpy as np

for batch_no in range(1,6):
    d 	= pickle.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'.format(batch_no), 'rb'))
    d2 	= json.load(open('bioasq_all/bioasq7/document_results/test_batch_{}/jpdrmm.json'.format(batch_no,batch_no)))
    id2rel = {}
    for q in d2['questions']:
        id2rel[q['id']] = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in q['documents']]
    recalls = []
    for quer in d['queries']:
        retr_ids = [doc['doc_id'] for doc in quer['retrieved_documents'][:10] if doc['doc_id'] in id2rel[quer['query_id']]]
        recalls.append(float(len(retr_ids))/float(len(id2rel[quer['query_id']])))
    print(np.average(recalls))
    # break




