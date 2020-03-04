
__author__ = 'Dimitris'

import  json
import  numpy                       as np
from    tqdm                        import tqdm
import  pickle, os
from adhoc_vectorizer import get_sentence_vecs
from my_sentence_splitting import get_sents

in_dir = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_1/bioasq8_bm25_top100/'

docs_data   = os.path.join(in_dir, )
ret_data    = os.path.join(in_dir, )




