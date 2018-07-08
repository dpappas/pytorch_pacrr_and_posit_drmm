

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Embedding, Conv1D, Input, LeakyReLU, Lambda, Concatenate, Dense
from keras.layers import Add, TimeDistributed, PReLU, GlobalAveragePooling1D
from keras.models import Model

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

doc1_       = np.random.randint(0,vocab_size, (1000, 500))
doc2_       = np.random.randint(0,vocab_size, (1000, 500))
quest_      = np.random.randint(0,vocab_size, (1000, 200))
labels      = np.random.randint(0,2,(1000))

H           = model.fit([doc1_,quest_],labels,validation_data=None, epochs=5,verbose=1,batch_size=32)








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






