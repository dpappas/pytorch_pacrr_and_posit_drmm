

import json
from pprint import pprint
import cPickle as pickle

dataloc = '/home/dpappas/for_ryan/'

print('loading pickle data')
with open(dataloc +'BioASQ-trainingDataset6b.json', 'r') as f:
    bioasq6_data = json.load(f)
    bioasq6_data = dict( (q['id'], q) for q in bioasq6_data['questions'] )

pprint(bioasq6_data['58b548d722d3005309000005'])

for item in bioasq6_data.values():
    for doc in item['documents']:
        if('snippets' in item):
            t = [sn for sn in item['snippets'] if(sn['document']==doc)]
        else:
            t = []
        if(len(t)==0):
            print '{} {}'.format(item['id'], doc.split('/')[-1])

# with open(dataloc + 'bioasq_bm25_top100.test.pkl', 'rb') as f:
#     test_data = pickle.load(f)
# with open(dataloc + 'bioasq_bm25_docset_top100.test.pkl', 'rb') as f:
#     test_docs = pickle.load(f)
# with open(dataloc + 'bioasq_bm25_top100.dev.pkl', 'rb') as f:
#     dev_data = pickle.load(f)
# with open(dataloc + 'bioasq_bm25_docset_top100.dev.pkl', 'rb') as f:
#     dev_docs = pickle.load(f)
# with open(dataloc + 'bioasq_bm25_top100.train.pkl', 'rb') as f:
#     train_data = pickle.load(f)
# with open(dataloc + 'bioasq_bm25_docset_top100.train.pkl', 'rb') as f:
#     train_docs = pickle.load(f)
# print('loading words')
#













