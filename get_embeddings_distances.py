


from gensim.models import KeyedVectors

w2v_bin_path        = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
wv_from_bin = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)







