
import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


glove_fpath = '/media/dpappas/dpappas_data/EMBEDDINGS/glove.6B.50d.txt'
w2v_fpath   = "/media/dpappas/dpappas_data/EMBEDDINGS/word2vec.6B.50d.txt"

if os.path.exists(w2v_fpath):
    tmp_file = get_tmpfile(w2v_fpath)
else:
    glove_file = datapath(glove_fpath)
    tmp_file = get_tmpfile(w2v_fpath)
    glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)
print(model.wv['this is a test'.split()].shape)


from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from pprint import pprint
fpath = '/home/DATA/Biomedical/other/BiomedicalWordEmbeddings/binary/biomedical-vectors-200.bin'
wv_from_bin = KeyedVectors.load_word2vec_format(datapath(fpath), binary=True)  # C bin format
token1 = 'KATP'.lower()
token2 = 'potassium'.lower()
pprint(wv_from_bin.most_similar(positive=[token1]))
print(wv_from_bin.similarity(token1, token2))

