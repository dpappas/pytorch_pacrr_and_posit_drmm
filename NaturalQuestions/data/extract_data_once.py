
from elasticsearch  import Elasticsearch
from elasticsearch.helpers import scan
from tqdm import tqdm
from pprint import pprint
from bs4 import BeautifulSoup
import re, nltk
from difflib import SequenceMatcher
import pickle

bioclean_mod    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()

stopwords       = nltk.corpus.stopwords.words("english")
elk_ip          = '192.168.188.80'

def clean_start_end(word):
    word = re.sub(r'(^\W+)', r'\1 ', word)
    word = re.sub(r'(\W+$)', r' \1', word)
    word = re.sub(r'\s+', ' ', word)
    return word.strip()

def tokenize(text):
    ret = []
    for token in nltk.tokenize.word_tokenize(text):
        ret.extend(clean_start_end(token).split())
    return ret

def get_first_n(question, n):
    question    = bioclean_mod(question)
    question    = [t for t in question if t not in stopwords]
    question    = ' '.join(question)
    ################################################
    doc_index   = 'natural_questions_0_1'
    es          = Elasticsearch(['{}:9200'.format(elk_ip)], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    ################################################
    bod         = {"size": n, "query": {"match": {"paragraph_text": question}}}
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

def get_all_quests():
    ################################################
    questions_index = 'natural_questions_q_0_1'
    questions_map   = "natural_questions_q_map_0_1"
    es              = Elasticsearch(['{}:9200'.format(elk_ip)], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    bod             = {}
    items           = scan(es, query=bod, index=questions_index, doc_type=questions_map)
    total           = es.count(index=questions_index, body=bod)['count']
    ################################################
    return items, total

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

train_quests, total_train_quests = get_all_quests()
my_counter  = 0
pbar        = tqdm(train_quests, total=total_train_quests)
zero_count  = 0
#################
train_data  = []
dev_data    = []
#################
for quest in pbar:
    qtext           = quest['_source']['question']
    short_answer    = quest['_source']['short_answer']
    long_answer     = BeautifulSoup(quest['_source']['long_answer'], 'lxml').text.strip()
    ####################
    if ('<table>' in quest['_source']['long_answer'].lower()):
        continue
    ####################
    all_retr_docs   = get_first_n(qtext, 100)
    ####################
    relevant_docs, irrelevant_docs = [], []
    for ret_doc in all_retr_docs:
        paragraph_text  = ' '.join(tokenize(ret_doc['_source']['paragraph_text']))
        ############################################
        if(short_answer in ret_doc['_source']['paragraph_text']):
            similarity = similar(paragraph_text, long_answer)
            if(similarity > 0.8 ):
                relevant_docs.append(ret_doc)
            else:
                irrelevant_docs.append(ret_doc)
        else:
            irrelevant_docs.append(ret_doc)
    if(len(relevant_docs)==0):
        zero_count += 1
    ####################
    quest['_source']['relevant_docs']   = relevant_docs
    quest['_source']['irrelevant_docs'] = irrelevant_docs
    ####################
    if(quest['_source']['dataset'] == 'train'):
        train_data.append(quest)
    else:
        dev_data.append(quest)
    ####################
    pbar.set_description('{} - {}'.format(zero_count, total_train_quests))

# pickle.dump(train_data, open('/home/dpappas/NQ_data/NQ_train_data.pkl', 'wb'), protocol=2)
# pickle.dump(dev_data,   open('/home/dpappas/NQ_data/NQ_dev_data.pkl',   'wb'), protocol=2)
pickle.dump(train_data, open('/home/dpappas/NQ_data/NQ_train_data_new.pkl', 'wb'), protocol=2)
pickle.dump(dev_data,   open('/home/dpappas/NQ_data/NQ_dev_data_new.pkl',   'wb'), protocol=2)

