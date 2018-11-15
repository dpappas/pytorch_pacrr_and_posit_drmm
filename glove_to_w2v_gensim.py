

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/media/dpappas/dpappas_data/EMBEDDINGS/glove.6B.50d.txt')
tmp_file = get_tmpfile("/media/dpappas/dpappas_data/EMBEDDINGS/word2vec.6B.50d.txt")
glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)




