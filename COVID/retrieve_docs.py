

import sys, os, re, json, pickle, ijson
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np
from elasticsearch import Elasticsearch
from pprint import pprint
import torch

#####################################################################################
# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()
bioclean        = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
#####################################################################################
with open('/home/dpappas/bioasq_all/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

print(stopwords)
#####################################################################################

my_seed     = 1989
use_cuda    = torch.cuda.is_available()
if(use_cuda):
    torch.cuda.manual_seed(my_seed)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu   = torch.cuda.device_count()

#####################################################################################
es          = Elasticsearch(['127.0.0.1:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
doc_index   = 'covid_index_0_1'

def tokenize(x):
  return bioclean(x)

def get_first_n_1(qtext, n, max_year=2020):
    tokenized_body  = bioclean_mod(qtext)
    tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(tokenized_body)
    print(question)
    ################################################
    bod             = {
        "size": n,
        "query": {
            "bool": {
                "must": [{"range": {"date": {"gte": "1800", "lte": str(max_year), "format": "dd/MM/yyyy||yyyy"}}}],
                "should": [{"match": {"joint_text": {"query": question, "boost": 1}}}],
                "minimum_should_match": 1,
            }
        }
    }
    res = es.search(index=doc_index, body=bod, request_timeout=120)
    ################################################
    results = res['hits']['hits']
    #######################################################
    temp_1 = {
        'num_rel': 0,
        'num_rel_ret': 0,
        'num_ret': -1,
        'query_id': 1234567890,
        'query_text': qtext,
        'relevant_documents': [],
        'retrieved_documents': []
    }
    #######################################################
    all_scores = [res['_score'] for res in results]
    print(all_scores)
    scaler = StandardScaler().fit(np.array(all_scores).reshape(-1, 1))
    temp_1['num_ret'] = len(all_scores)
    #######################################################
    for res, rank in zip(results, range(1, len(results) + 1)):
        temp_1['retrieved_documents'].append({
            'bm25_score'        : res['_score'],
            'doc_id'            : res['_id'],
            'is_relevant'       : False,
            'norm_bm25_score'   : scaler.transform([[res['_score']]])[0][0],
            'rank'              : rank,
            'doc'               : {
                'title'             : res['_source']['joint_text'].split('------------------------------', 1)[0].strip(),
                'abstractText'      : res['_source']['joint_text'].split('------------------------------', 1)[1].strip(),
                'doi'               : res['_source']['doi'],
                'pmcid'             : res['_source']['pmcid'],
                'pmid'              : res['_source']['pmid'],
                'section'           : res['_source']['section'],
                'date'              : res['_source']['date']
            }
        })
    return temp_1

# pprint(
#     get_first_n_1(
#         qtext       = 'A pneumonia outbreak associated with a new coronavirus of probable bat origin',
#         n           = 100,
#         max_year    = 2021
#     )
# )

#####################################################################################

'''

from pytorch_transformers import BertModel, BertTokenizer

def encode_sent_with_bert(sent, max_len = 512):
    sent_ids        = [bert_tokenizer.encode(sent, add_special_tokens=True)[:max_len]]
    _, sent_vec     = bert_model(torch.LongTensor(sent_ids).to(device))
    return sent_vec

scibert_dir     = '/home/dpappas/scibert_scivocab_uncased'
bert_tokenizer  = BertTokenizer.from_pretrained(scibert_dir)
bert_model      = BertModel.from_pretrained(scibert_dir,  output_hidden_states=False, output_attentions=False).to(device)

text    = 'A pneumonia outbreak associated with a new coronavirus of probable bat origin'
vec     = encode_sent_with_bert(text, max_len=512)[0].cpu().detach().numpy().tolist()

body = {
      "_source": ["joint_text"],
      "query": {
        "script_score": {
          "query" : {
            "match" : {
                "joint_text" : {
                    "query" : text
                }
            }
          },
          "script": {
            "source": 'params.vec_weight * (cosineSimilarity(params.query_vector, "doc_vec_scibert") + 1.0) + params.bm25_weight * _score',
            "params": {
                "query_vector": vec,
                "vec_weight": 10.0,
                "bm25_weight": 1.0
            }
          }
        }
      }
}

res = es.search(index=index, body=body)
pprint(res)

'''

#####################################################################################
