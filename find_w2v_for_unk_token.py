from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import re
import numpy as np

softmax  	= lambda z: np.exp(z) / np.sum(np.exp(z))
bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def get_for_window(tokens_in_window):
    h       = np.array([model.wv.word_vec(t) for t in tokens_in_window])
    h       = np.mean(h, axis=0)
    z       = np.matmul(model.wv.syn0, h)
    y       = softmax(z)
    vec     = np.matmul( y.reshape((1,-1)), model.wv.syn0).squeeze()
    ret     = model.similar_by_vector(vec, 5).squeeze()
    return ret

model       = KeyedVectors.load_word2vec_format(datapath("/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin"), binary=True)
sentence    = 'The demand for inpatient and ICU beds for COVID-19 in the US: lessons from Chinese cities'
tokens      = bioclean(sentence)

for i in range(len(tokens)):
    tok = tokens[i]
    try:
        model[tok]
    except KeyError:
        print(tok)
        print(get_for_window(tokens[i-3:i]+tokens[i+1:i+3]))













