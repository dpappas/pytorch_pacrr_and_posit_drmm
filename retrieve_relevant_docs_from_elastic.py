

import json
import cPickle as pickle
from pprint import pprint

bioasq_data_path    = '/home/DATA/Biomedical/bioasq6/bioasq6_data/BioASQ-trainingDataset6b.json'
data                = json.load(open(bioasq_data_path, 'r'))
ddd                 = {}
for quest in data['questions']:
    if('snippets' in quest):
        for sn in quest['snippets']:
            pmid    = sn['document'].split('/')[-1]
            ttt     = sn['text'].strip()
            bod     = quest['body'].strip()
            if(bod not in ddd):
                ddd[bod] = {}
            if(pmid not in ddd[bod]):
                ddd[bod][pmid] = [ttt]
            else:
                ddd[bod][pmid].append(ttt)

abs_path    = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_docset_top100.train.pkl'
all_abs     = pickle.load(open(abs_path,'rb'))



# fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_top100.full_train.pkl'
# fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_docset_top100.train.pkl'
# fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_top100.train.pkl'
# fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_docset_top100.full_train.pkl'
# fpath   = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_top100.dev.pkl'
# fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_docset_top100.dev.pkl'

d1      = data['queries'][0]
pprint(d1)

data = pickle.load(open(fpath,'rb'))
pprint(data['21177106'])

from bioasq_utils import get_sents, similar
get_sents(data['24787386']['abstractText'])

f2 = '/home/DATA/Biomedical/bioasq6/bioasq6_data/'
f3 = '/home/DATA/Biomedical/bioasq6/bioasq6_data/'










