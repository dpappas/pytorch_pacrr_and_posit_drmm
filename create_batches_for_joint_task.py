#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import re
import operator
import gensim
UNK_TOKEN = '*UNK*'

def map_term2ind(w2v_path):
    word_vectors        = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    vocabulary          = sorted(list(word_vectors.vocab.keys()))
    term2ind            = dict([t[::-1] for t in enumerate(vocabulary, start=1)])
    term2ind[UNK_TOKEN] = max(term2ind.items(), key=operator.itemgetter(1))[1] + 1	# Index of *UNK* token
    print('Size of voc: {0}'.format(len(vocabulary)))
    print('Unknown terms\'s id: {0}'.format(term2ind['*UNK*']))
    return term2ind

w2v_path = '/home/DATA/Biomedical/bioasq6/bioasq6_data/word_embeddings/pubmed2018_w2v_200D.bin'
term2ind = map_term2ind(w2v_path)

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

all_data = pickle.load(open('joint_task_data.p','rb'))
print(len(all_data))






