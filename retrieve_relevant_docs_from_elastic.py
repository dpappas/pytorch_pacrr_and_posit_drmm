

import json
import cPickle as pickle
from pprint import pprint

def preprocess_bioasq_data(bioasq_data_path):
    data = json.load(open(bioasq_data_path, 'r'))
    ddd = {}
    for quest in data['questions']:
        if ('snippets' in quest):
            for sn in quest['snippets']:
                pmid = sn['document'].split('/')[-1]
                ttt = sn['text'].strip()
                bod = quest['body'].strip()
                if (bod not in ddd):
                    ddd[bod] = {}
                if (pmid not in ddd[bod]):
                    ddd[bod][pmid] = [ttt]
                else:
                    ddd[bod][pmid].append(ttt)
    return ddd

bioasq_data_path    = '/home/DATA/Biomedical/bioasq6/bioasq6_data/BioASQ-trainingDataset6b.json'
ddd                 = preprocess_bioasq_data(bioasq_data_path)

abs_path    = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.train.pkl'
all_abs     = pickle.load(open(abs_path,'rb'))

fpath = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.train.pkl'

d1      = data['queries'][0]
pprint(d1)


from bioasq_utils import get_sents, similar
get_sents(data['24787386']['abstractText'])

f2 = '/home/DATA/Biomedical/bioasq6/bioasq6_data/'
f3 = '/home/DATA/Biomedical/bioasq6/bioasq6_data/'










