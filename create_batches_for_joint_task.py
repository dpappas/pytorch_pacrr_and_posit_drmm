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
    UNK_TOKEN           = '*UNK*'
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
    print('Loaded w2v vectors with gensim')
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
    matrix          = np.stack(matrix, 0)
    #
    print('Size of voc: {0}'.format(len(vocabulary)))
    print('Unknown terms\'s id: {0}'.format(t2i['UNKN']))
    return t2i, i2t, matrix

w2v_path = '/home/DATA/Biomedical/bioasq6/bioasq6_data/word_embeddings/pubmed2018_w2v_200D.bin'
t2i, i2t, matrix = load_w2v_embs(w2v_path)

all_data = pickle.load(open('joint_task_data.p','rb'))
print(len(all_data))

max_nof_sents       = max([len(item['all_sents']) for item in all_data])
max_len_of_sents    = 400 # basika einai 355 sta train
non_sent            = [0] * max_len_of_sents
si, sy = [], []
for item in all_data:
    indices     = [[t2i[t] for t in bioclean(s)] for s in item['all_sents']]
    indices     = [s + ([0] * (max_len_of_sents - len(s))) for s in indices]
    indices     = indices + ((max_nof_sents - len(indices)) * [non_sent])
    #
    sent_inds   = np.array(indices)
    sent_y      = np.array(item['sent_sim_vec'] + ((max_nof_sents - len(item['sent_sim_vec'])) * [0]))
    #
    print(sent_inds.shape)
    print(sent_y.shape)
    #
    # 'question', 'all_sents', 'sent_sim_vec', 'doc_rel'








