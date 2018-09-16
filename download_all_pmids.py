
import json
import cPickle as pickle
from pprint import pprint

def load_all_data(dataloc):
    print('loading pickle data')
    #
    with open(dataloc+'BioASQ-trainingDataset6b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict((q['id'], q) for q in bioasq6_data['questions'])
    with open(dataloc + 'bioasq_bm25_top100.test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.dev.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    print('loading words')
    #
    return test_data, dev_data, train_data, bioasq6_data

def get_all_ids_from_data(test_data, dev_data, train_data, bioasq6_data):
    all_ids = []
    for quer in train_data['queries']+dev_data['queries']+test_data['queries']:
        all_ids.extend([rd['doc_id'] for rd in quer['retrieved_documents']])
    for val in bioasq6_data.values():
        all_ids.extend([d.split('/')[-1] for d in val['documents']])
        if('snippets' in val):
            all_ids.extend([sn['document'].split('/')[-1] for sn in val['snippets']])
    all_ids = list(set(all_ids))
    return all_ids

w2v_bin_path    = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
idf_pickle_path = '/home/dpappas/for_ryan/fordp/idf.pkl'
dataloc         = '/home/dpappas/for_ryan/'

(test_data, dev_data, train_data, bioasq6_data) = load_all_data(dataloc=dataloc)

all_ids = get_all_ids_from_data(test_data, dev_data, train_data, bioasq6_data)

