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

def get_sim_matrix_one(qtoks, stoks, max_len_of_quests, max_len_of_sents):
    sm = np.zeros((max_len_of_quests,max_len_of_sents))
    for i in range(len(qtoks)):
        for j in range(len(stoks)):
            if(qtoks[i] == stoks[j]):
                sm[i,j] = 1.
    return sm

def get_sim_matrix(item, max_len_of_quests, max_len_of_sents, max_nof_sents):
    qtoks       = bioclean(item['question'])
    sim_matrix  = []
    for s in item['all_sents']:
        stoks   = bioclean(s)
        sim_matrix.append(get_sim_matrix_one(qtoks, stoks, max_len_of_quests, max_len_of_sents))
    sim_matrix.extend((max_nof_sents - len(sim_matrix)) * [np.zeros((max_len_of_quests, max_len_of_sents))])
    sim_matrix = np.stack(sim_matrix,0)
    return sim_matrix

w2v_path = '/home/DATA/Biomedical/bioasq6/bioasq6_data/word_embeddings/pubmed2018_w2v_200D.bin'
t2i, i2t, matrix = load_w2v_embs(w2v_path)

all_data = pickle.load(open('joint_task_data.p','rb'))
print(len(all_data))

odir = '/home/dpappas/joint_task_batches/'
if not os.path.exists(odir):
    os.makedirs(odir)

b_size              = 16
qi, dy, si, sy, sm  = [], [], [], [], []
metr_batch          = 0
for item in tqdm(all_data):
    indices     = [[ get_index(token, t2i) for token in bioclean(s)] for s in item['all_sents']]
    quest_inds  = [ get_index(token, t2i) for token in bioclean(item['question'])]

        #
        sent_inds   = np.array(indices)
        #
        sent_y      = np.array(item['sent_sim_vec'] + ((max_nof_sents - len(item['sent_sim_vec'])) * [0]))
        #
        quest_inds  = quest_inds + ([0] * (max_len_of_quests - len(quest_inds)))
        quest_inds  = np.array(quest_inds)
        #
        all_sims = get_sim_matrix(item, max_len_of_quests, max_len_of_sents, max_nof_sents)
        #
        qi.append(quest_inds)
        si.append(sent_inds)
        sy.append(sent_y)
        dy.append(item['doc_rel'])
        sm.append(all_sims)
        if(len(dy) == b_size):
            si, sy,qi, dy, sm = np.stack(si, 0), np.stack(sy, 0), np.stack(qi, 0), np.stack(dy, 0), np.stack(sm, 0)
            metr_batch += 1
            pickle.dump(
                {
                    'sent_inds'     : si,
                    'sent_labels'   : sy,
                    'quest_inds'    : qi,
                    'doc_labels'    : dy,
                    'sim_matrix'    : sm
                },
                open(odir+'{}.p'.format(metr_batch),'wb')
            )
            qi, dy, si, sy, sm = [], [], [], [], []
    #
