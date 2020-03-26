

from elasticsearch import Elasticsearch
from pytorch_transformers import BertModel, BertTokenizer
from pprint import pprint
import torch

my_seed     = 1989
use_cuda    = torch.cuda.is_available()
if(use_cuda):
    torch.cuda.manual_seed(my_seed)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu   = torch.cuda.device_count()

def encode_sent_with_bert(sent, max_len = 512):
    sent_ids        = [bert_tokenizer.encode(sent, add_special_tokens=True)[:max_len]]
    _, sent_vec     = bert_model(torch.LongTensor(sent_ids).to(device))
    return sent_vec

scibert_dir     = '/home/dpappas/scibert_scivocab_uncased'
bert_tokenizer  = BertTokenizer.from_pretrained(scibert_dir)
bert_model      = BertModel.from_pretrained(scibert_dir,  output_hidden_states=False, output_attentions=False).to(device)

es      = Elasticsearch(['127.0.0.1:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
index   = 'covid_index_0_1'

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

