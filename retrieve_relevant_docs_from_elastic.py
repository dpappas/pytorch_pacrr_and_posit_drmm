

import cPickle as pickle
from pprint import pprint
# fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_top100.full_train.pkl'
# fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_docset_top100.train.pkl'
# fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_top100.train.pkl'
# fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_docset_top100.full_train.pkl'

fpath   = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_top100.dev.pkl'
data    = pickle.load(open(fpath,'rb'))
d1      = data['queries'][0]
pprint(d1)

fpath = '/home/DATA/Biomedical/bioasq6/bioasq6_data/bioasq6_bm25_top100/bioasq6_bm25_docset_top100.dev.pkl'
data = pickle.load(open(fpath,'rb'))
pprint(data['21177106'])

from bioasq_utils import get_sents, similar
get_sents(data['24787386']['abstractText'])

