
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from pprint import pprint
import re, pickle

def check_if_doc_exists(document_title):
    bod = {
        "size": 1,
        "query": {
            "bool": {
                "must": [
                    {"term": {"document_title.raw": document_title}}
                ]
            }
        }
    }
    res = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['total'] > 0

def get_first_n(question, n):
    question    = bioclean_mod(question)
    question    = [t for t in question if t not in stopwords]
    question    = ' '.join(question)
    print(question)
    bod = {
        "size": n,
        "query": {"match": {"paragraph_text": question}}
    }
    res = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

bioclean_mod = lambda t: re.sub(
    '[.,?;*!%^&_+():-\[\]{}]', '',
    t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()
).split()

################################################
questions_index = 'natural_questions_q_0_1'
questions_map   = "natural_questions_q_map_0_1"
################################################
doc_index       = 'natural_questions_0_1'
doc_map         = "natural_questions_map_0_1"
################################################
es          = Elasticsearch(['192.168.188.80:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
items       = scan(es, query=None, index=questions_index, doc_type=questions_map)
################################################
with open('stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)
################################################

for item in items:
    pprint(item)
    print(20 * '-')
    if(check_if_doc_exists(item['_source']['document_title'])):
        pprint(get_first_n(item['_source']['question'], 100))
    exit()


