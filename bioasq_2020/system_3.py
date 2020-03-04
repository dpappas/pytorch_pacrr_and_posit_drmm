
__author__ = 'Dimitris'

import  numpy as np
from    tqdm  import tqdm
import  pickle, os, json
from adhoc_vectorizer import get_sentence_vecs
from my_sentence_splitting import get_sents
from pprint import pprint

in_dir      = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_1/bioasq8_bm25_top100/'

docs_data   = pickle.load(open(os.path.join(in_dir, 'bioasq8_bm25_docset_top100.test.pkl'), 'rb'))
ret_data    = pickle.load(open(os.path.join(in_dir, 'bioasq8_bm25_top100.test.pkl'), 'rb'))

pprint(ret_data['queries'][0])

for quer in ret_data['queries']:
    qid     = quer['query_id']
    qtext   = quer['query_text']



