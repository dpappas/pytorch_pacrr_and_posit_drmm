

from gensim.models import KeyedVectors

w2v_bin_path_old                = '/home/dpappas/COVID/COVID/pubmed2018_w2v_30D.bin'
w2v_bin_path_new                = '/home/dpappas/COVID/covid_19_w2v_embeds_30.model'
wv_old = KeyedVectors.load_word2vec_format(w2v_bin_path_old, binary=True)
wv_new = KeyedVectors.load_word2vec_format(w2v_bin_path_new, binary=True)


