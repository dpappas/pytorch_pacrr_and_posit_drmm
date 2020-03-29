

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models import translation_matrix
from gensim.models import BackMappingTranslationMatrix
from pprint import pprint

w2v_bin_path_old    = '/home/dpappas/COVID/COVID/pubmed2018_w2v_30D.bin'
w2v_bin_path_new    = '/home/dpappas/COVID/covid_19_w2v_embeds_30.model'
wv_old              = KeyedVectors.load_word2vec_format(w2v_bin_path_old, binary=True)
wv_new              = Word2Vec.load(w2v_bin_path_new)

common_tokens       = set(wv_old.vocab.keys()).intersection(set(wv_new.wv.vocab.keys()))
common_tokens       = [(tok, tok) for tok in common_tokens]

transmat = translation_matrix.TranslationMatrix(wv_new.wv, wv_old, common_tokens)
transmat.train(common_tokens)

# transmat.apply_transmat(transmat.source_space)

pprint(transmat.translate('covid-19', topn=25))

from scipy import spatial
result = 1 - spatial.distance.cosine(wv_old['fredriksberg'], wv_old['non-neurologist'])
