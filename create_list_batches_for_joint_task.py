#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import gensim
import operator
import numpy as np
from tqdm import tqdm
import cPickle as pickle
from pprint import pprint
import os

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

def get_index(token, t2i):
    try:
        return t2i[token]
    except KeyError:
        return t2i['UNKN']

def get_sim_mat(stoks, qtoks):
    sm = np.zeros((len(stoks), len(qtoks)))
    for i in range(len(qtoks)):
        for j in range(len(stoks)):
            if(qtoks[i] == stoks[j]):
                sm[j,i] = 1.
    return sm

w2v_path = '/home/DATA/Biomedical/bioasq6/bioasq6_data/word_embeddings/pubmed2018_w2v_200D.bin'
t2i, i2t, matrix = load_w2v_embs(w2v_path)
b_size  = 32

all_data = pickle.load(open('joint_task_data.p','rb'))
print(len(all_data))

odir    = '/home/dpappas/joint_task_list_batches/train/'
if not os.path.exists(odir):
    os.makedirs(odir)

qi, dy, si, sy, sm  = [], [], [], [], []
metr_batch          = 0
for item in tqdm(all_data):
    sents_inds  = [[get_index(token, t2i) for token in bioclean(s)] for s in item['all_sents']]
    quest_inds  = [get_index(token, t2i) for token in bioclean(item['question'])]
    all_sims    = [get_sim_mat(stoks, quest_inds) for stoks in sents_inds]
    sent_y      = np.array(item['sent_sim_vec'])
    #
    qi.append(quest_inds)
    si.append(sents_inds)
    sy.append(sent_y)
    dy.append(item['doc_rel'])
    sm.append(all_sims)
    #
    if(len(dy) == b_size):
        metr_batch          += 1
        pickle.dump({'sent_inds':si, 'sent_labels':sy, 'quest_inds':qi, 'doc_labels':dy, 'sim_matrix':sm}, open(odir+'{}.p'.format(metr_batch),'wb'))
        qi, dy, si, sy, sm  = [], [], [], [], []
    #


odir = '/home/dpappas/joint_task_list_batches/dev/'
if not os.path.exists(odir):
    os.makedirs(odir)


