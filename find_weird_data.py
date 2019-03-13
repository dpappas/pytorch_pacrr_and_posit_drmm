

import json
from pprint import pprint
import pickle

dataloc = '/home/dpappas/for_ryan/'

print('loading pickle data')
with open(dataloc +'BioASQ-trainingDataset6b.json', 'r') as f:
    bioasq6_data = json.load(f)
    bioasq6_data = dict((q['id'], q) for q in bioasq6_data['questions'])

pprint(bioasq6_data['58b548d722d3005309000005'])

for item in bioasq6_data.values():
    for doc in item['documents']:
        if('snippets' in item):
            t = [sn for sn in item['snippets'] if(sn['document']==doc)]
        else:
            t = []
        if(len(t)==0):
            print('{} {}'.format(item['id'], doc.split('/')[-1]))

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


# from gensim.models.keyedvectors import KeyedVectors
# w2v_bin_path    = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
# wv = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
# min_tok, min_count = None, None
# for word in wv.vocab.items():
#     if(word[1].count < 3):
#         print word[0], word[1].count
#     if(min_tok is None):
#         min_tok     =  word[0]
#         min_count   =  word[1].count
#     else:
#         if(min_count) >= word[1].count:
#             min_tok     =  word[0]
#             min_count   =  word[1].count
#
# print(min_tok, min_count)










