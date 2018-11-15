
import os
import json
from pprint import pprint
import cPickle as pickle
from gensim.models import KeyedVectors

def load_all_data(dataloc):
    print('loading json data')
    #
    train_data  = json.load(open(os.path.join(dataloc,'train-v1.1.json')))['data']
    print len(train_data)
    pprint(train_data[0]['title'])
    pprint(train_data[0]['paragraphs'][0])
    exit()
    dev_data    = json.load(open(os.path.join(dataloc,'dev-v1.1.json')))['data']
    print len(dev_data)
    test_data   = json.load(open(os.path.join(dataloc,'dev-v1.1.json')))['data']
    print len(test_data)
    exit()
    #
    # words           = {}
    # GetWords(train_data, train_docs, words)
    # GetWords(dev_data,   dev_docs,   words)
    # GetWords(test_data,  test_docs,  words)
    # mgmx
    # print('loading idfs')
    # idf, max_idf    = load_idfs(idf_pickle_path, words)
    print('loading w2v')
    w2v_txt_path    = os.path.join(dataloc, 'word2vec.6B.50d.txt')
    wv              = KeyedVectors.load_word2vec_format(w2v_txt_path, binary=False)
    wv              = dict([(word, wv[word]) for word in wv.vocab.keys() if(word in words)])
    # return test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, wv, bioasq6_data

# -v1.1.json
# /home/dpappas/for_ryan/squad_ir_data/'
dataloc     = '/home/dpappas/for_ryan/squad_ir_data/'
load_all_data(dataloc)





