



import os, re, json, keras
import numpy as np
from tqdm import tqdm
from pprint import pprint
from nltk.tokenize import sent_tokenize
from gensim.models.keyedvectors import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
from keras.layers import (Bidirectional, Dense, Embedding, GRU, Input, TimeDistributed, GaussianNoise, Dropout, Lambda, Concatenate)
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from attention import Attention
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import Counter
from keras import backend as K

my_seed = 1
import random
random.seed(my_seed)
from numpy.random import seed
seed(my_seed)
from tensorflow import set_random_seed
set_random_seed(my_seed)

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def first_alpha_is_upper(sent):
    specials = [
        '__EU__','__SU__','__EMS__','__SMS__','__SI__',
        '__ESB','__SSB__','__EB__','__SB__','__EI__',
        '__EA__','__SA__','__SQ__','__EQ__','__EXTLINK',
        '__XREF','__URI', '__EMAIL','__ARRAY','__TABLE',
        '__FIG','__AWID','__FUNDS'
    ]
    for special in specials:
        sent = sent.replace(special,'')
    for c in sent:
        if(c.isalpha()):
            if(c.isupper()):
                return True
            else:
                return False
    return False

def ends_with_special(sent):
    sent = sent.lower()
    ind = [item.end() for item in re.finditer('[\W\s]sp.|[\W\s]nos.|[\W\s]figs.|[\W\s]sp.[\W\s]no.|[\W\s][vols.|[\W\s]cv.|[\W\s]fig.|[\W\s]e.g.|[\W\s]et[\W\s]al.|[\W\s]i.e.|[\W\s]p.p.m.|[\W\s]cf.|[\W\s]n.a.', sent)]
    if(len(ind)==0):
        return False
    else:
        ind = max(ind)
        if (len(sent) == ind):
            return True
        else:
            return False

def split_sentences2(text):
    sents = [l.strip() for l in sent_tokenize(text)]
    ret = []
    i = 0
    while (i < len(sents)):
        sent = sents[i]
        while (
            ((i + 1) < len(sents)) and
            (
                ends_with_special(sent)        or
                not first_alpha_is_upper(sents[i+1])
                # sent[-5:].count('.') > 1       or
                # sents[i+1][:10].count('.')>1   or
                # len(sent.split()) < 2          or
                # len(sents[i+1].split()) < 2
            )
        ):
            sent += ' ' + sents[i + 1]
            i += 1
        ret.append(sent.replace('\n',' ').strip())
        i += 1
    return ret

def get_sents(ntext):
    sents = []
    for subtext in ntext.split('\n'):
        subtext = re.sub( '\s+', ' ', subtext.replace('\n',' ') ).strip()
        if (len(subtext) > 0):
            ss = split_sentences2(subtext)
            sents.extend([ s for s in ss if(len(s.strip())>0)])
    if(len(sents[-1]) == 0 ):
        sents = sents[:-1]
    return sents

def get_embedding_layer(embeddings, max_sent_length):
    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        weights=[embeddings],
        input_length=max_sent_length,
        mask_zero=True,
        trainable=False)
    return embedding_layer

def hrnn_title_abstract(embeddings, max_sents, max_sent_length, total_classes, **kwargs):
    ######################################################
    # HyperParameters
    ######################################################
    drop_input      = kwargs.get("drop_input", 0.2)
    noise_input     = kwargs.get("noise_input", 0.2)
    rnn_size        = kwargs.get("rnn_size", 150)
    #
    rnn_rec_drop    = kwargs.get("rnn_rec_drop", 0)
    rnn_drop        = kwargs.get("rnn_drop", 0.3)
    att_drop        = kwargs.get("att_drop", 0.3)
    #
    #######################
    # WORDS RNN
    #######################
    # define input
    sentence_input  = Input(shape=(max_sent_length,), dtype='int32')
    #
    # embed words, using an Embedding layer
    embedding_layer = get_embedding_layer(embeddings, max_sent_length)
    words           = embedding_layer(sentence_input)
    #
    # Regularize embedding layer:
    # - add gaussian noise to word vectors
    words           = GaussianNoise(noise_input)(words)
    # - add dropout to word vectors
    words           = Dropout(drop_input)(words)
    #
    # read each sentence, which is a sequence of words vectors and generate a fixed vector representation.
    h_words         = Bidirectional(GRU(rnn_size, return_sequences=True, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop))(words)
    sentence        = Attention()(h_words)
    sentence        = Dropout(att_drop)(sentence)
    #
    sent_encoder    = Model(sentence_input, sentence)
    print(sent_encoder.summary())
    #
    #######################
    # SENTENCE RNN
    #######################
    document_input  = Input(shape=(max_sents, max_sent_length, ), dtype='float32')
    document_enc    = TimeDistributed(sent_encoder)(document_input)
    #
    # Now we have a single vector representation for each sentence.
    # Next just like before, we want to feed the vector of each sentence,
    # the sequence of sentence vectors, to another RNN, which will generate
    # a fixed vector representation for the whole document.
    h_sentences     = Bidirectional(GRU(rnn_size, return_sequences=True, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop))(document_enc)
    #
    #######################
    # CLASSIFIER
    #######################
    output_layer    = Dense(total_classes, activation='softmax', activity_regularizer=l2(.0001))
    preds           = TimeDistributed(output_layer)(h_sentences)
    #
    model           = Model(document_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=5), metrics=['acc', f1])
    #
    return model

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def vectorize_sent(text, word2idx, max_length):
    words = np.zeros(max_length).astype(int)
    # trim tokens after max length
    text = text[:max_length]
    for i, token in enumerate(text):
        if token in word2idx:
            words[i] = word2idx[token]
        else:
            words[i] = word2idx["<unk>"]
    return words

def vectorize_doc(doc, word2idx, max_sents, max_length):
    # trim sentences after max_sents
    doc  = doc[:max_sents]
    _doc = np.zeros((max_sents, max_length), dtype='int32')
    for i, sent in enumerate(doc):
        s = vectorize_sent(sent, word2idx, max_length)
        _doc[i] = s
    return _doc

def get_all_pubs_paths(folder):
    ret = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            pp = os.path.join(root, file)
            if (pp.endswith('.json') and 'pubmed_pubs' in pp):
                ret.append(pp)
    return ret

def do_for_sents(sents):
    splitted_sents  = [bioclean(s) for s in sents]
    input_feats     = vectorize_doc(splitted_sents, w2i, MAX_SENTS, MAX_SENT_LENGTH)
    preds           = model.predict(np.stack([input_feats], 0))[0]
    pred_class      = np.argmax(preds, axis=1)
    ret             = [{'type': tt[0], 'text': tt[1]} for tt in zip(class_encoder.inverse_transform(pred_class), sents)]
    return ret

class_enc_name      = '/home/dpappas/sections_classification/pubsec_class_encoder_5classes.npy'
w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
checkpoint_fpath    = '/home/dpappas/weights-improvement-5classes-20-0.98.hdf5'

class_encoder           = LabelEncoder()
class_encoder.classes_  = np.load(class_enc_name)
print(class_encoder.classes_)

wv                  = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
embeds              = [np.zeros(wv.vector_size), np.average(wv.vectors, 0)]
w2i                 = {}
w2i['<pad>']        = 0
w2i['<unk>']        = 1
for w in tqdm(sorted(wv.vocab.keys())):
    w2i[w] = len(w2i)
    embeds.append(wv[w])

embeddings      = np.stack(embeds)
EMB_SIZE        = 30
MAX_SENTS       = 30
MAX_SENT_LENGTH = 50
batch_size      = 1
total_classes   = len(class_encoder.classes_)
model           = hrnn_title_abstract(embeddings, MAX_SENTS, MAX_SENT_LENGTH, total_classes)

print(model.summary())

model.load_weights(checkpoint_fpath)


