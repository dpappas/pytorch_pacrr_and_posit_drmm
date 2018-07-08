
import re
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import random
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
                    yield good_sents_inds, good_all_sims, bad_sents_inds, bad_all_sims, bad_quest_inds

def load_data():
    print('Loading abs texts...')
    logger.info('Loading abs texts...')
    train_all_abs = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.train.pkl', 'rb'))
    dev_all_abs = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.dev.pkl', 'rb'))
    test_all_abs = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.test.pkl', 'rb'))
    print('Loading retrieved docsc...')
    logger.info('Loading retrieved docsc...')
    train_bm25_scores = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.train.pkl', 'rb'))
    dev_bm25_scores = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.dev.pkl', 'rb'))
    test_bm25_scores = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.test.pkl', 'rb'))
    print('Loading token to index files...')
    logger.info('Loading token to index files...')
    token_to_index_f = '/home/dpappas/joint_task_list_batches/t2i.p'
    t2i = pickle.load(open(token_to_index_f, 'rb'))
    print('yielding data')
    logger.info('yielding data')
    return train_all_abs, dev_all_abs, test_all_abs, train_bm25_scores, dev_bm25_scores, test_bm25_scores, t2i

train_all_abs, dev_all_abs, test_all_abs, train_bm25_scores, dev_bm25_scores, test_bm25_scores, t2i = load_data()



def myGenerator():
    for f in fs:
        d = pickle.load(open(f,'rb'))
        yield d['x'],d['y']

pad_sequences(xs, maxlen=story_maxlen)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
            self,
            list_IDs,
            labels,
            batch_size=32,
            dim=(32,32,32),
            n_channels=1,
            n_classes=10,
            shuffle=True
    ):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load('data/' + ID + '.npy')
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

k = 5

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

embedding_weights   = np.load('/home/dpappas/joint_task_list_batches/embedding_matrix.npy')
# embedding_weights = np.random.rand(100,20)
vocab_size          = embedding_weights.shape[0]
emb_size            = embedding_weights.shape[1]


doc1                = Input(shape=(500,), dtype='int32')
quest               = Input(shape=(200,), dtype='int32')
emb_layer           = Embedding(vocab_size, emb_size, weights=[embedding_weights])
d1_embeds           = emb_layer(doc1)
q_embeds            = emb_layer(quest)
trigram_conv        = Conv1D(emb_size, 3, padding="same")
conv_activation     = LeakyReLU()
d1_trigrams         = conv_activation(trigram_conv(d1_embeds))
q_trigrams          = conv_activation(trigram_conv(q_embeds))
d1_trigrams         = Add()([d1_trigrams, d1_embeds])
q_trigrams          = Add()([q_trigrams, q_embeds])
sim_insens_d1       = Lambda(pairwise_cosine_sim)([q_embeds, d1_embeds])
sim_sens_d1         = Lambda(pairwise_cosine_sim)([q_embeds, d1_trigrams])
pooled_d1_insens    = Lambda(average_k_max_pool)(sim_insens_d1)
pooled_d1_sens      = Lambda(average_k_max_pool)(sim_sens_d1)
concated_d1         = Concatenate()([pooled_d1_insens, pooled_d1_sens])
hidden              = Dense(8, activation=LeakyReLU())
hd1                 = TimeDistributed(hidden)(concated_d1)
out_layer           = Dense(1, activation=LeakyReLU())
od1                 = TimeDistributed(out_layer)(hd1)
od1                 = GlobalAveragePooling1D()(od1)

model               = Model(inputs=[doc1, quest], outputs=od1)
model.compile(optimizer='sgd', loss='hinge', metrics=['accuracy'])

doc1_               = np.random.randint(0,vocab_size, (1000, 500))
doc2_               = np.random.randint(0,vocab_size, (1000, 500))
quest_              = np.random.randint(0,vocab_size, (1000, 200))
labels              = np.random.randint(0,2,(1000))

H                   = model.fit([doc1_,quest_],labels,validation_data=None, epochs=5,verbose=1,batch_size=32)








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






