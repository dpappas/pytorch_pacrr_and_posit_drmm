
import numpy as np
from gensim.models import KeyedVectors
w2v_path        = '/home/DATA/Biomedical/other/BiomedicalWordEmbeddings/binary/biomedical-vectors-200.bin'
word_vectors    = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

#
vocabulary      = sorted(list(word_vectors.vocab.keys()))
vecs            = []
vecs            = np.array([word_vectors[word] for word in vocabulary])
#
av_vec          = np.average(vecs, axis=0)
pad_vec         = np.zeros(200)
noise1          = np.random.normal(0, 1, 200)
noise2          = np.random.normal(0, 1, 200)
qunkn_vec       = av_vec + noise1
dunkn_vec       = av_vec + noise2

all_vecs        = np.concatenate([np.expand_dims(pad_vec,0), np.expand_dims(qunkn_vec,0), np.expand_dims(dunkn_vec,0), vecs], axis=0)
all_voc         = ['PAD', 'QUNKN', 'DUNKN'] + vocabulary







