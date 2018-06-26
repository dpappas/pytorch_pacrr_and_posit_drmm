#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import gensim
import operator
import numpy as np
from tqdm import tqdm
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

def get_index(token, t2i):
    try:
        return t2i[token]
    except KeyError:
        return t2i['UNKN']

def get_sim_matrix_one(qtoks, stoks, max_len_of_sents):
    sm = np.zeros((max_len_of_sents,max_len_of_sents))
    for i in range(len(qtoks)):
        for j in range(len(stoks)):
            if(qtoks[i] == stoks[j]):
                sm[i,j] = 1.
    return sm

def get_sim_matrix(item, max_len_of_sents, max_nof_sents):
    qtoks       = bioclean(item['question'])
    sim_matrix  = []
    for s in item['all_sents']:
        stoks   = bioclean(s)
        sim_matrix.append(get_sim_matrix_one(qtoks, stoks, max_len_of_sents))
    sim_matrix.extend((max_nof_sents - len(sim_matrix)) * [np.zeros((max_len_of_sents, max_len_of_sents))])
    sim_matrix = np.stack(sim_matrix,0)
    return sim_matrix

w2v_path = '/home/DATA/Biomedical/bioasq6/bioasq6_data/word_embeddings/pubmed2018_w2v_200D.bin'
t2i, i2t, matrix = load_w2v_embs(w2v_path)

all_data = pickle.load(open('joint_task_data.p','rb'))
print(len(all_data))

max_nof_sents       = max([len(item['all_sents']) for item in all_data])
max_len_of_sents    = 400 # basika einai 355 sta train
non_sent            = [0] * max_len_of_sents
qi, dy, si, sy      = [], [], [], []
for item in tqdm(all_data):
    indices     = [[ get_index(token, t2i) for token in bioclean(s)] for s in item['all_sents']]
    indices     = [s + ([0] * (max_len_of_sents - len(s))) for s in indices]
    indices     = indices + ((max_nof_sents - len(indices)) * [non_sent])
    sent_inds   = np.array(indices)
    #
    sent_y      = np.array(item['sent_sim_vec'] + ((max_nof_sents - len(item['sent_sim_vec'])) * [0]))
    #
    quest_inds  = [ get_index(token, t2i) for token in bioclean(item['question'])]
    quest_inds  = quest_inds + ([0] * (max_len_of_sents - len(quest_inds)))
    quest_inds  = np.array(quest_inds)
    #
    all_sims = get_sim_matrix(item, max_len_of_sents, max_nof_sents)
    #
    # print(all_sims.shape)
    # print(sent_inds.shape)
    # print(sent_y.shape)
    # print(quest_inds.shape)
    # print(all_sims.sum())
    # print(20 * '-')
    #


'''
non_sent            = [0] * max_len_of_sents
qi, dy, si, sy      = [], [], [], []
for item in tqdm(all_data):
    indices     = [[ get_index(token, t2i) for token in bioclean(s)] for s in item['all_sents']]
    indices     = [s + ([0] * (max_len_of_sents - len(s))) for s in indices]
    indices     = indices + ((max_nof_sents - len(indices)) * [non_sent])
    sent_inds   = np.array(indices)
    #
    sent_y      = np.array(item['sent_sim_vec'] + ((max_nof_sents - len(item['sent_sim_vec'])) * [0]))
    #
    quest_inds  = [ get_index(token, t2i) for token in bioclean(item['question'])]
    quest_inds  = quest_inds + ([0] * (max_len_of_sents - len(quest_inds)))
    quest_inds  = np.array(quest_inds)
    #
    qi.append(quest_inds)
    si.append(sent_inds)
    sy.append(sent_y)
    dy.append(item['doc_rel'])
    #
    # print(sent_inds.shape)
    # print(sent_y.shape)
    # print(quest_inds.shape)
    #
    # 'question', 'all_sents', 'sent_sim_vec', 'doc_rel'

si = np.stack(si, 0)
sy = np.stack(sy, 0)
qi = np.stack(qi, 0)
dy = np.stack(dy, 0)

print(si.shape)
print(sy.shape)
print(qi.shape)
print(dy.shape)

pickle.dump(
    {
        'sents'         : si,
        'quest'         : qi,
        'sent_labels'   : sy,
        'doc_labels'    : dy
    },
    open('entire_train_joint_dataset.p', 'wb')
)

'''







