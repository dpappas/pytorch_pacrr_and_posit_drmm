
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from pprint import pprint
fpath = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
wv_from_bin = KeyedVectors.load_word2vec_format(datapath(fpath), binary=True)  # C bin format
token1 = 'KATP'.lower()
token2 = 'potassium'.lower()
pprint(wv_from_bin.most_similar(positive=[token1]))
print(wv_from_bin.similarity(token1, token2))


