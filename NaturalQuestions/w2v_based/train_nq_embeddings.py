
from    tqdm            import tqdm
import  gensim, os
from    gensim.models import FastText
import  logging
from    pprint          import pprint
import re
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

def clean_start_end(word):
    word = re.sub(r'(^\W+)', r'\1 ', word)
    word = re.sub(r'(\W+$)', r' \1', word)
    word = re.sub(r'\s+', ' ', word)
    return word.strip()

def my_tokenize(text):
    ret = []
    for token in word_tokenize(text):
        ret.extend(clean_start_end(token).split())
    return ret

class MakeIter(object):
    def __init__(self, generator_func):
        self.generator_func     = generator_func
    def __iter__(self):
        for item in self.generator_func():
            yield item

def yield_lines():
    ################################################
    es      = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    items   = scan(es, query=None, index=doc_index, doc_type=doc_map)
    total = es.count(index=doc_index)['count']
    ################################################
    for item in tqdm(items, total=total):
        sents = sent_tokenize(item['_source']['paragraph_text'])
        # pprint(sents)
        for sent in sents:
            tokens = [t.lower() for t in word_tokenize(sent)]
            # pprint(tokens)
            yield tokens

################################################
doc_index       = 'natural_questions_0_1'
doc_map         = "natural_questions_map_0_1"
################################################

iterator1       = MakeIter(yield_lines)

total_examples  = 0
for item in iterator1.__iter__():
    total_examples += 1

iterator1       = MakeIter(yield_lines)

################################################

size            = 30
model           = gensim.models.Word2Vec(iterator1, size=size, window=10, min_count=4, workers=10)
model.train(iterator1, total_examples=total_examples, epochs=20)
model.save("lower_nq_w2v_{}.model".format(size))

################################################

'''

from    pprint                      import pprint
import gensim

print('loading w2v')
w2v_path    = '/home/dpappas/NQ_data/lower_nq_w2v_30.model'
wv          = gensim.models.Word2Vec.load(w2v_path)
wv          = dict([(word, wv[word]) for word in wv.wv.vocab.keys()])
# wv              = dict([(word, wv[word]) for word in wv.vocab.keys() if (word in words)])

'''

