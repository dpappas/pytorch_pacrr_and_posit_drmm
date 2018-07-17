
import re
import keras
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
from keras.layers import Embedding, Conv1D, Input, LeakyReLU, Lambda, Concatenate, Dense
from keras.layers import Add, TimeDistributed, PReLU, GlobalAveragePooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

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

def get_item_inds(item, question, t2i):
    passage     = item['title'] + ' ' + item['abstractText']
    all_sims    = get_sim_mat(bioclean(passage), bioclean(question))
    sents_inds  = [get_index(token, t2i) for token in bioclean(passage)]
    quest_inds  = [get_index(token, t2i) for token in bioclean(question)]
    return sents_inds, quest_inds, all_sims

def data_yielder(bm25_scores, all_abs, t2i, how_many_loops):
    for quer in bm25_scores[u'queries']:
        quest       = quer['query_text']
        ret_pmids   = [t[u'doc_id'] for t in quer[u'retrieved_documents']]
        good_pmids  = [t for t in ret_pmids if t in quer[u'relevant_documents']]
        bad_pmids   = [t for t in ret_pmids if t not in quer[u'relevant_documents']]
        if(len(bad_pmids)>0):
            for i in range(how_many_loops):
                for gid in good_pmids:
                    bid                                             = random.choice(bad_pmids)
                    good_sents_inds, good_quest_inds, good_all_sims = get_item_inds(all_abs[gid], quest, t2i)
                    bad_sents_inds, bad_quest_inds, bad_all_sims    = get_item_inds(all_abs[bid], quest, t2i)
                    yield good_sents_inds,  good_all_sims,  good_quest_inds,    1
                    yield bad_sents_inds,   bad_all_sims,   bad_quest_inds,     0

def load_data():
    print('Loading abs texts...')
    train_all_abs   = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.train.pkl', 'rb'))
    dev_all_abs     = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.dev.pkl', 'rb'))
    test_all_abs    = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.test.pkl', 'rb'))
    print('Loading retrieved docsc...')
    train_bm25_scores   = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.train.pkl', 'rb'))
    dev_bm25_scores     = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.dev.pkl', 'rb'))
    test_bm25_scores    = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.test.pkl', 'rb'))
    print('Loading token to index files...')
    token_to_index_f = '/home/dpappas/joint_task_list_batches/t2i.p'
    t2i                 = pickle.load(open(token_to_index_f, 'rb'))
    print('yielding data')
    return train_all_abs, dev_all_abs, test_all_abs, train_bm25_scores, dev_bm25_scores, test_bm25_scores, t2i

def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """
    square_sum  = K.sum(K.square(x), axis=axis, keepdims=True)
    norm        = K.sqrt(K.maximum(square_sum, K.epsilon()))
    return norm

def pairwise_cosine_sim(A_B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """
    A, B    = A_B
    A_mag   = l2_norm(A, axis=2)
    B_mag   = l2_norm(B, axis=2)
    num     = K.batch_dot(A, K.permute_dimensions(B, (0,2,1)))
    den     = (A_mag * K.permute_dimensions(B_mag, (0,2,1)))
    dist_mat =  num / den
    return dist_mat

def average_k_max_pool(inputs):
    top_k           = tf.nn.top_k(inputs, k=k, sorted=True, name=None)[0]
    average_top_k   = tf.reduce_mean(top_k,axis=-1)
    average_top_k   = tf.expand_dims(average_top_k, axis=-1)
    maxim           = tf.reduce_max(inputs,axis=-1)
    maxim           = tf.expand_dims(maxim, axis=-1)
    concatenated    = tf.concat([maxim, average_top_k],axis=-1)
    return concatenated

def myGenerator(bm25_scores, all_abs, t2i, how_many_loops, story_maxlen, b_size):
    x1, x2, y = [], [], []
    for sents_inds, _, quest_inds, label in data_yielder(bm25_scores, all_abs, t2i, how_many_loops):
        x1.append(sents_inds)
        x2.append(quest_inds)
        y.append(label)
        if(len(y) == b_size):
            yield [
                      np.array(pad_sequences(x1, maxlen=story_maxlen)),
                      np.array(pad_sequences(x2, maxlen=story_maxlen))
                  ], np.array(y)
            x1, x2, y = [], [], []

def the_objective(negatives_positives):
    negatives, positives = negatives_positives
    margin               = 1.0
    delta                = negatives - positives
    loss_q_pos           = tf.reduce_sum(tf.nn.relu(margin + delta), axis=-1)
    return loss_q_pos

story_maxlen = 500

# train_all_abs, dev_all_abs, test_all_abs, train_bm25_scores, dev_bm25_scores, test_bm25_scores, t2i = load_data()

# d = myGenerator(train_bm25_scores, train_all_abs, t2i, 1, story_maxlen=story_maxlen, b_size=32)
# aa = d.next()

k = 5

# embedding_weights   = np.load('/home/dpappas/joint_task_list_batches/embedding_matrix.npy')
embedding_weights   = np.random.rand(100,20)
vocab_size          = embedding_weights.shape[0]
emb_size            = embedding_weights.shape[1]


quest               = Input(shape=(story_maxlen,), dtype='int32')
doc1                = Input(shape=(story_maxlen,), dtype='int32')
doc2                = Input(shape=(story_maxlen,), dtype='int32')
#
hidden              = Dense(8, activation=LeakyReLU())
out_layer           = Dense(1, activation=None)
#
emb_layer           = Embedding(vocab_size, emb_size, weights=[embedding_weights])
q_embeds            = emb_layer(quest)
trigram_conv        = Conv1D(emb_size, 3, padding="same")
conv_activation     = LeakyReLU()
q_trigrams          = conv_activation(trigram_conv(q_embeds))
q_trigrams          = Add()([q_trigrams, q_embeds])
#
d1_embeds           = emb_layer(doc1)
d1_trigrams         = conv_activation(trigram_conv(d1_embeds))
d1_trigrams         = Add()([d1_trigrams, d1_embeds])
sim_insens_d1       = Lambda(pairwise_cosine_sim)([q_embeds, d1_embeds])
sim_sens_d1         = Lambda(pairwise_cosine_sim)([q_embeds, d1_trigrams])
pooled_d1_insens    = Lambda(average_k_max_pool)(sim_insens_d1)
pooled_d1_sens      = Lambda(average_k_max_pool)(sim_sens_d1)
concated_d1         = Concatenate()([pooled_d1_insens, pooled_d1_sens])
hd1                 = TimeDistributed(hidden)(concated_d1)
od1                 = TimeDistributed(out_layer)(hd1)
od1                 = GlobalAveragePooling1D()(od1)
#
d2_embeds           = emb_layer(doc2)
d2_trigrams         = conv_activation(trigram_conv(d2_embeds))
d2_trigrams         = Add()([d2_trigrams, d2_embeds])
sim_insens_d2       = Lambda(pairwise_cosine_sim)([q_embeds, d2_embeds])
sim_sens_d2         = Lambda(pairwise_cosine_sim)([q_embeds, d2_trigrams])
pooled_d2_insens    = Lambda(average_k_max_pool)(sim_insens_d2)
pooled_d2_sens      = Lambda(average_k_max_pool)(sim_sens_d2)
concated_d2         = Concatenate()([pooled_d2_insens, pooled_d2_sens])
hd2                 = TimeDistributed(hidden)(concated_d2)
od2                 = TimeDistributed(out_layer)(hd2)
od2                 = GlobalAveragePooling1D()(od2)
#
the_loss            = Lambda(the_objective)([od2, od1])
#
model               = Model(inputs=[doc1, doc2, quest], outputs=the_loss)
model.compile(optimizer='adam', loss='mae')

doc1_               = np.random.randint(0,vocab_size, (1000, story_maxlen))
doc2_               = np.random.randint(0,vocab_size, (1000, story_maxlen))
quest_              = np.random.randint(0,vocab_size, (1000, story_maxlen))
# labels              = np.random.randint(0,2,(1000))
labels              = np.zeros((1000,1))

H = model.fit(
    [doc1_, doc2_, quest_],
    labels,
    validation_data=None,
    epochs=5,
    verbose=1,
    batch_size=32
)

# H                   = model.fit_generator(
#     myGenerator(train_bm25_scores, train_all_abs, t2i, 1, story_maxlen=1500, b_size=32),
#     steps_per_epoch  = 10000,
#     epochs           = 5,
#     validation_data  = None,
#     validation_steps = None,
# )




'''


# doc2                = Input(shape=(500,), dtype='int32')
# d2_embeds           = emb_layer(doc2)
# d2_trigrams         = conv_activation(trigram_conv(d2_embeds))
# sim_insens_d2       = Lambda(pairwise_cosine_sim)([q_embeds, d2_embeds])
# sim_sens_d2         = Lambda(pairwise_cosine_sim)([q_embeds, d2_trigrams])
# pooled_d2_insens    = Lambda(average_k_max_pool)(sim_insens_d2)
# pooled_d2_sens      = Lambda(average_k_max_pool)(sim_sens_d2)
# concated_d2         = Concatenate()([pooled_d2_insens, pooled_d2_sens])
# hd2                 = TimeDistributed(hidden)(concated_d2)
# od2                 = TimeDistributed(out_layer)(hd2)
# od2                 = GlobalAveragePooling1D()(od2)
# print(od2.get_shape())

doc1_    = np.random.randint(0,vocab_size, (1000, 500))
doc2_    = np.random.randint(0,vocab_size, (1000, 500))
quest_   = np.random.randint(0,vocab_size, (1000, 200))
labels   = [np.ones((1000)), np.zeros((1000))]

H = model.fit(
    [ doc1_, doc2_, quest_ ],
    labels,
    validation_data   = None,
    epochs            = 5,
    verbose           = 1,
    batch_size        = 32
)



'''






