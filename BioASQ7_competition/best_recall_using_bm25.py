
import pickle, json
import numpy as np
batch_no = 4
d 	= pickle.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'.format(batch_no), 'rb'))
d2 	= json.load(open('bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(batch_no,batch_no)))

id2rel = {}
for q in d2['questions']:
	id2rel[q['id']] = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in q['documents']]

# recalls = []
# for quer in d['queries']:
# 	retr_ids = [doc['doc_id'] for doc in quer['retrieved_documents'] if doc['doc_id'] in id2rel[quer['query_id']]]
# 	recalls.append(float(len(retr_ids))/float(len(id2rel[quer['query_id']])))
#
# np.average(recalls)

recalls = []
for quer in d['queries']:
	retr_ids = [doc['doc_id'] for doc in quer['retrieved_documents'][:10] if doc['doc_id'] in id2rel[quer['query_id']]]
	recalls.append(float(len(retr_ids))/float(len(id2rel[quer['query_id']])))

np.average(recalls)
