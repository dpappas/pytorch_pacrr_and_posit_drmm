
import os
import re
import json
import keras
import random
import subprocess
import numpy as np
import tensorflow as tf
import keras.backend as K
import cPickle as pickle
from tqdm import tqdm
from keras.layers import Embedding, Conv1D, Input, LeakyReLU, Lambda, Concatenate, Dense
from keras.layers import Add, TimeDistributed, PReLU, GlobalAveragePooling1D, multiply
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from my_bioasq_preprocessing import get_item_inds
from pprint import pprint
from keras.callbacks import ModelCheckpoint
from my_bioasq_preprocessing import get_item_inds, text2indices, get_sim_mat
from my_bioasq_preprocessing import bioclean, get_overlap_features_mode_1

def random_data_yielder(bm25_scores, all_abs, t2i, how_many):
    while(how_many>0):
        quer        = random.choice(bm25_scores[u'queries'])
        quest       = quer['query_text']
        bm25s       = {t['doc_id']:t['norm_bm25_score'] for t in quer[u'retrieved_documents']}
        ret_pmids   = [t[u'doc_id'] for t in quer[u'retrieved_documents']]
        good_pmids  = [t for t in ret_pmids if t in quer[u'relevant_documents']]
        bad_pmids   = [t for t in ret_pmids if t not in quer[u'relevant_documents']]
        if(len(bad_pmids)>0 and len(good_pmids)>0):
            how_many -= 1
            gid = random.choice(good_pmids)
            bid = random.choice(bad_pmids)
            good_sents_inds, good_quest_inds, good_all_sims, additional_features_good   = get_item_inds(all_abs[gid], quest, t2i)
            additional_features_good.append(bm25s[gid])
            bad_sents_inds, bad_quest_inds, bad_all_sims, additional_features_bad       = get_item_inds(all_abs[bid], quest, t2i)
            additional_features_bad.append(bm25s[bid])
            yield good_sents_inds, good_all_sims, bad_sents_inds, bad_all_sims, bad_quest_inds, np.array(additional_features_good), np.array(additional_features_bad)

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

def myGenerator(bm25_scores, all_abs, t2i, story_maxlen, quest_maxlen, b_size):
    gsi, gas, bsi, bas, qis, gafs, bafs = [], [], [], [], [], [], []
    for good_sents_inds, good_all_sims, bad_sents_inds, bad_all_sims, quest_inds, gaf, baf in random_data_yielder(
        bm25_scores, all_abs, t2i, 150000
    ):
        gsi.append(good_sents_inds)
        bsi.append(bad_sents_inds)
        qis.append(quest_inds)
        gas.append(good_all_sims)
        bas.append(bad_all_sims)
        gafs.append(gaf)
        bafs.append(baf)
        if(len(gsi) == b_size):
            yield [
                      np.array(pad_sequences(gsi, maxlen=story_maxlen)),
                      np.array(pad_sequences(bsi, maxlen=story_maxlen)),
                      np.array(pad_sequences(qis, maxlen=quest_maxlen)),
                      np.array(gafs),
                      np.array(bafs)
            ], np.zeros((len(gsi),1))
            gsi, gas, bsi, bas, qis, gafs, bafs = [], [], [], [], [], [], []

def the_objective(negatives_positives):
    margin               = 1.0
    negatives, positives = negatives_positives
    delta                = negatives - positives
    loss_q_pos           = tf.reduce_sum(tf.nn.relu(margin + delta), axis=-1)
    loss_q_pos           = tf.reshape(loss_q_pos,(-1,1))
    return loss_q_pos

def compute_doc_output(doc, q_embeds, q_trigrams, weights, doc_af, doc_mask):
    d_embeds        = emb_layer(doc)
    d_trigrams      = trigram_conv(d_embeds)
    d_trigrams      = conv_activ(d_trigrams)
    d_trigrams      = Add()([d_trigrams, d_embeds])
    sim_insens_d    = Lambda(pairwise_cosine_sim)([q_embeds, d_embeds])
    sim_insens_d    = multiply([sim_insens_d, doc_mask])
    sim_sens_d      = Lambda(pairwise_cosine_sim)([q_trigrams, d_trigrams])
    sim_sens_d      = multiply([sim_sens_d, doc_mask])
    pooled_d_insens = Lambda(average_k_max_pool)(sim_insens_d)
    pooled_d_sens   = Lambda(average_k_max_pool)(sim_sens_d)
    concated_d      = Concatenate()([pooled_d_insens, pooled_d_sens])
    hd              = TimeDistributed(hidden1)(concated_d)
    hd              = TimeDistributed(h1_activ)(hd)
    iod             = TimeDistributed(hidden2)(hd)
    iod             = TimeDistributed(h2_activ)(iod)
    iod             = multiply([iod, weights])
    iod             = GlobalAveragePooling1D()(iod)
    concated_iod_af = Concatenate()([iod, doc_af])
    od              = out_layer(concated_iod_af)
    return od

def process_question(quest):
    q_embeds        = emb_layer(quest)
    q_idfs          = idf_layer(quest)
    q_trigrams      = trigram_conv(q_embeds)
    q_trigrams      = conv_activ(q_trigrams)
    q_trigrams      = Add()([q_trigrams, q_embeds])
    weight_input    = Concatenate()([q_trigrams, q_idfs])
    weights         = weights_layer(weight_input)
    weights         = w_activ(weights)
    return q_embeds, q_trigrams, weights

def compute_masking(quest_doc):
    quest, doc  = quest_doc
    quest       = tf.to_float(quest>0)
    quest       = tf.reshape(quest, (-1, quest.shape[1], 1))
    doc         = tf.to_float(doc>0)
    doc         = tf.reshape(doc, (-1, doc.shape[1], 1))
    res         = K.batch_dot(quest, K.permute_dimensions(doc, (0, 2, 1)))
    return res

def get_map_res(fgold, femit):
    trec_eval_res   = subprocess.Popen(['python', '/home/DATA/Biomedical/document_ranking/eval/run_eval.py', fgold, femit], stdout=subprocess.PIPE, shell=False)
    (out, err)      = trec_eval_res.communicate()
    lines           = out.decode("utf-8").split('\n')
    map_res         = [l for l in lines if (l.startswith('map '))][0].split('\t')
    map_res         = float(map_res[-1])
    return map_res

def test_one(doc1_, quest_, doc1_af_):
    get_doc_emit = K.function(model.inputs, [od1])
    test_inputs  = [doc1_, doc1_, quest_, doc1_af_, doc1_af_]
    out          = get_doc_emit(test_inputs)
    return out

def get_one_map(prefix, bm25_scores, all_abs):
    data = {}
    data['questions'] = []
    for quer in tqdm(bm25_scores['queries']):
        dato        = {'body': quer['query_text'], 'id': quer['query_id'], 'documents': []}
        bm25s       = {t['doc_id']: t['bm25_score'] for t in quer[u'retrieved_documents']}
        doc_res     = {}
        quest_inds  = text2indices(quer['query_text'], t2i, 'q')
        gsi, gas, qis, gafs, docs_ids = [], [], [], [], []
        for retr in quer['retrieved_documents']:
            doc_id          = retr['doc_id']
            passage         = all_abs[doc_id]['title'] + ' ' + all_abs[doc_id]['abstractText']
            sents_inds      = text2indices(passage, t2i, 'd')
            gaf             = get_overlap_features_mode_1(bioclean(quer['query_text']), bioclean(passage))
            gaf.append(bm25s[doc_id])
            #
            gsi.append(sents_inds)
            qis.append(quest_inds)
            gafs.append(gaf)
            docs_ids.append(doc_id)
        #
        gsi             = pad_sequences(gsi,        maxlen=story_maxlen)
        qis             = pad_sequences(quest_inds, maxlen=quest_maxlen)
        gafs            = np.array(gafs)
        doc1_emit_      = test_one(gsi, qis, gafs)
        for i in range(len(docs_ids)):
            doc_res[docs_ids[i]] = doc1_emit_[i]
        #
        doc_res             = sorted(doc_res.items(), key=lambda x: x[1], reverse=True)
        doc_res             = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
        doc_res             = doc_res[:100]
        dato['documents']   = doc_res
        data['questions'].append(dato)
    if (prefix == 'dev'):
        with open(odir + 'elk_relevant_abs_posit_drmm_lists_dev.json', 'w') as f:
            f.write(json.dumps(data, indent=4, sort_keys=True))
        res_map = get_map_res(
            '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq.dev.json',
            odir + 'elk_relevant_abs_posit_drmm_lists_dev.json'
        )
    else:
        with open(odir + 'elk_relevant_abs_posit_drmm_lists_test.json', 'w') as f:
            f.write(json.dumps(data, indent=4, sort_keys=True))
        res_map = get_map_res(
            '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq.test.json',
            odir + 'elk_relevant_abs_posit_drmm_lists_test.json'
        )
    return res_map

odir = '/home/dpappas/simplest_posit_drmm_keras/'
if not os.path.exists(odir):
    os.makedirs(odir)

story_maxlen = 1500
quest_maxlen = 100

train_all_abs, dev_all_abs, test_all_abs, train_bm25_scores, dev_bm25_scores, test_bm25_scores, t2i = load_data()
k = 5

embedding_weights   = np.load('/home/dpappas/joint_task_list_batches/embedding_matrix.npy')
idf_weights         = np.load('/home/dpappas/joint_task_list_batches/idf_matrix.npy')
# embedding_weights   = np.random.rand(100,20)
# idf_weights         = np.random.rand(100)
idf_weights         = idf_weights.reshape((-1, 1))
vocab_size          = embedding_weights.shape[0]
emb_size            = embedding_weights.shape[1]

quest               = Input(shape=(quest_maxlen,), dtype='int32')
doc1                = Input(shape=(story_maxlen,), dtype='int32')
doc2                = Input(shape=(story_maxlen,), dtype='int32')
doc1_af             = Input(shape=(4,), dtype='float32')
doc2_af             = Input(shape=(4,), dtype='float32')
#
emb_layer           = Embedding(vocab_size, emb_size,   weights=[embedding_weights], trainable=False)
idf_layer           = Embedding(vocab_size, 1,          weights=[idf_weights],       trainable=False)
trigram_conv        = Conv1D(emb_size, 3, padding="same")
conv_activ          = LeakyReLU()
hidden1             = Dense(8)
h1_activ            = LeakyReLU()
hidden2             = Dense(1)
h2_activ            = LeakyReLU()
weights_layer       = Dense(1)
w_activ             = LeakyReLU()
out_layer           = Dense(1)
#
doc1_mask                       = Lambda(compute_masking)([quest, doc1])
doc2_mask                       = Lambda(compute_masking)([quest, doc2])
q_embeds, q_trigrams, weights   = process_question(quest)
od1                             = compute_doc_output(doc1, q_embeds, q_trigrams, weights, doc1_af, doc1_mask)
od2                             = compute_doc_output(doc2, q_embeds, q_trigrams, weights, doc2_af, doc2_mask)
#
the_loss            = Lambda(the_objective)([od2, od1])
#
model               = Model(inputs=[doc1, doc2, quest, doc1_af, doc2_af], outputs=the_loss)
optimizer           = keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

# quest_              = np.random.randint(0, vocab_size, (1000, quest_maxlen))
# doc1_               = np.random.randint(0, vocab_size, (1000, story_maxlen))
# doc2_               = np.random.randint(0, vocab_size, (1000, story_maxlen))
# doc1_af_            = np.random.randn(1000, 4)
# doc2_af_            = np.random.randn(1000, 4)
# labels              = np.zeros((1000,1))
# H = model.fit([doc1_, doc2_, quest_, doc1_af_, doc2_af_], labels, validation_data=None, epochs=5, verbose=1, batch_size=32)

filepath        ="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint      = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list  = [checkpoint]

train_history   = model.fit_generator(
    generator           = myGenerator(train_bm25_scores, train_all_abs, t2i, story_maxlen, quest_maxlen, 32),
    steps_per_epoch     = 5, #100,
    epochs              = 3,  #30,
    validation_data     = myGenerator(dev_bm25_scores, dev_all_abs, t2i, story_maxlen, quest_maxlen, 32),
    validation_steps    = 2, #20,
    callbacks           = callbacks_list,
    verbose             = 1
)

test_map    = get_one_map('test', test_bm25_scores, test_all_abs)
print(test_map)

'''

class SaveTheModel(keras.callbacks.Callback):
    best_valid_loss = 10e5
    def on_epoch_end(self, epoch, logs):
        if(self.best_valid_loss>logs['val_loss']):
            print('saving model on epoch:{}'.format(epoch))
            self.best_valid_loss = logs['val_loss']
            open('keras_posit_drmm_model.json', 'w').write(model.to_json())
            model.save_weights('keras_posit_drmm_weights.h5')

callbacks_list = [SaveTheModel()]

'''

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

'''
Epoch 00001: val_loss improved from inf to 2.52525, saving model to weights-improvement-01-2.53.hdf5
'''




