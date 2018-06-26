#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import gensim
import operator
import numpy as np
import cPickle as pickle
from pprint import pprint

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def has_alnum(token):
    for c in token:
        if(c.isalnum()):
            return True
    return False

def map_term2ind(w2v_path):
    word_vectors        = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    vocabulary          = sorted(list(word_vectors.vocab.keys()))
    term2ind            = dict(
        [
            t[::-1]
            for t in enumerate(vocabulary, start=1)
        ]
    )
    term2ind[UNK_TOKEN] = max(term2ind.items(), key=operator.itemgetter(1))[1] + 1	# Index of *UNK* token
    print('Size of voc: {0}'.format(len(vocabulary)))
    print('Unknown terms\'s id: {0}'.format(term2ind['*UNK*']))
    return term2ind

def load_w2v_embs(w2v_path):
    word_vectors    = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    vocabulary      = sorted(list(word_vectors.vocab.keys()))
    #
    av              = np.average(word_vectors.vectors,0)
    pad             = np.zeros(av.shape)
    #
    vocabulary      = ['PAD', 'UNKN'] + vocabulary
    t2i             = dict(t[::-1] for t in  enumerate(vocabulary))
    i2t             = dict(enumerate(vocabulary))
    #
    matrix          = [word_vectors[t] for t in vocabulary[2:]]
    matrix          = [pad, av] + matrix
    #
    print('Size of voc: {0}'.format(len(vocabulary)))
    print('Unknown terms\'s id: {0}'.format(t2i['*UNK*']))
    return t2i, i2t, matrix

w2v_path = '/home/DATA/Biomedical/bioasq6/bioasq6_data/word_embeddings/pubmed2018_w2v_200D.bin'
t2i, i2t, matrix = load_w2v_embs(w2v_path)

all_data = pickle.load(open('joint_task_data.p','rb'))
print(len(all_data))

max_nof_sents = max([len(item['all_sents']) for item in all_data])


for item in all_data:
    if(len(item['all_sents']) > 50):
        pprint([[t2i[t] for t in bioclean(s)] for s in item['all_sents']])
        print item['sent_sim_vec']
        print(20*'-')

# len([ k for k in term2ind if(has_alnum(k)) ])


