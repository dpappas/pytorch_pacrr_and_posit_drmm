
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from pprint import pprint
import re, pickle
from nltk.util import ngrams

def get_ngrams(tokens, n):
    n_grams = ngrams(tokens, n)
    return [' '.join(grams) for grams in n_grams]

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
    print(20 * '-')
    bod = {
        "size": n,
        "query": {"match": {"paragraph_text": question}}
    }
    ##############
    # the_shoulds = []
    # for qt in question.split():
    #     the_shoulds.append(
    #         {
    #             "match": {
    #                 "paragraph_text": qt
    #             }
    #         }
    #     )
    # # for ngram in get_ngrams(question.split(), 2):
    # #     the_shoulds.append({"match": {"paragraph_text": ngram}})
    # #####
    # bod = {
    #     "query": {
    #         "bool": {
    #             "must": [
    #                 {
    #                     "match": {
    #                         "paragraph_text": question
    #                     }
    #                 }
    #             ],
    #             "should": the_shoulds,
    #             "minimum_should_match": len(question.split())-2,
    #         }
    #     }
    # }
    ##############
    pprint(bod)
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
es          = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
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


