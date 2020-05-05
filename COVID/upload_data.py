
import zipfile, json, hashlib, sys
from pprint import pprint
from tqdm import tqdm
from elasticsearch import Elasticsearch
import torch
from dateutil import parser
from pytorch_transformers import BertModel, BertTokenizer
from elasticsearch.helpers import bulk

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

def create_an_action(tw):
    tw['_op_type']  = 'index'
    tw['_index']    = index
    return tw

def send_to_elk(actions):
    flag = True
    while (flag):
        try:
            result = bulk(elastic_con, iter(actions))
            pprint(result)
            flag = False
        except Exception as e:
            print(e)
            if ('ConnectionTimeout' in str(e)):
                print('Retrying')
            else:
                flag = False

scibert_dir     = '/home/dpappas/scibert_scivocab_uncased'
bert_tokenizer  = BertTokenizer.from_pretrained(scibert_dir)
bert_model      = BertModel.from_pretrained(scibert_dir,  output_hidden_states=False, output_attentions=False).to(device)

# zip_path    = 'C:\\Users\\dvpap\\Downloads\\db_entries.zip'
# zip_path    = '/media/dpappas/dpappas_data/COVID/db_entries.zip'
# zip_path    = '/home/dpappas/db_entries.zip'
# zip_path = '/home/dpappas/COVID/data/28_03_2020/db_entries.zip'
# zip_path = '/home/dpappas/COVID/data/01_04_2020/db_new_entries.zip'
# zip_path = '/home/dpappas/COVID/data/12_04_2020/db_new_entries.zip'
# zip_path = '/home/dpappas/COVID/data/21_04_2020/db_new_entries.zip'
# zip_path = '/home/dpappas/COVID/data/04_05_2020/db_new_entries.zip'

fromm       = int(sys.argv[1])
too         = int(sys.argv[2])
zip_path    = sys.argv[3]

archive     = zipfile.ZipFile(zip_path, 'r')
d           = json.loads(archive.read('db_new_entries.json'))

index       = 'covid_index_0_1'
elastic_con = Elasticsearch(['127.0.01:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

# fromm       = 0
# too         = 10000000000000

actions     = []
b_size      = 200
print('total: {}'.format(len(d)))
for item in tqdm(d[fromm:too]):
    if(len(item['date'])==0):
        item['date']        = None
    else:
        try:
            item['date']    = parser.parse(item['date']).strftime('%Y-%m-%d')
        except:
            item['date']    = None
    # hash_object     = hashlib.md5(item['joint_text'].encode()).hexdigest()
    idd                     = '{}_{}_{}_{}'.format(item['pmid'], item['pmcid'], item['doi'], item['joint_text'][-50:])
    vec                     = encode_sent_with_bert(item['joint_text'].replace('\n------------------------------', ''), max_len=512)
    vec                     = vec[0].cpu().detach().numpy()
    item["doc_vec_scibert"] = vec.tolist()
    actions.append(create_an_action(item))
    if (len(actions) >= b_size):
        send_to_elk(actions)
        actions = []

send_to_elk(actions)

'''
python3.6 upload_data.py 0 130000 "/home/dpappas/COVID/data/04_05_2020/db_new_entries.zip"
python3.6 upload_data.py 130000 260000 "/home/dpappas/COVID/data/04_05_2020/db_new_entries.zip"

'''

'''

SEARCH LIKE

GET /covid_index_0_1/_search?pretty
{
  "_source": ["joint_text"],
  "query": {
    "script_score": {
      "query" : {
        "match" : {
            "joint_text" : {
                "query" : "table 5"
            }
        }
      },
      "script": {
        "source": "params.vec_weight * (cosineSimilarity(params.query_vector, \"doc_vec_scibert\") + 1.0) + params.bm25_weight * _score", 
        "params": {"query_vector": [0.32,-0.27,0.83,0.98,0.06,1.00,0.43,-1.00,0.25,-0.03,0.28,0.20,0.27,0.86,0.09,-0.06,-0.15,-0.21,-0.87,-1.00,0.12,0.38,0.55,0.27,-0.14,0.11,-0.99,-0.05,-0.27,0.32,0.39,-0.95,0.18,0.10,-0.03,0.11,1.00,0.26,0.22,-0.16,-0.27,0.23,0.25,-0.03,0.39,-0.16,-0.32,-0.05,-0.60,0.22,0.29,-0.12,-0.15,0.98,0.93,-0.04,0.01,0.27,-0.84,0.08,-0.36,-0.97,0.27,0.56,0.29,-0.33,-0.43,-0.27,0.12,0.32,1.00,0.40,0.32,-0.02,0.93,-0.08,-0.31,-0.00,0.20,0.06,0.47,-0.27,-0.22,-0.99,-0.23,-0.30,0.10,-0.08,0.99,-0.93,0.55,-0.23,-0.09,-0.26,0.24,-0.08,-0.11,-0.01,0.31,-0.40,0.06,0.27,0.33,-0.06,0.12,-0.09,0.02,-0.09,-0.04,0.46,-0.42,-0.12,0.12,-0.30,-0.21,-0.20,0.99,0.51,-0.08,0.05,0.54,0.92,0.97,-0.50,-0.26,-0.19,-0.36,0.54,0.11,0.28,-0.24,-0.65,-0.99,-0.25,0.11,0.05,0.10,-0.10,0.94,0.36,-0.46,-0.33,-0.12,0.33,-0.18,-0.97,0.94,0.23,0.41,0.39,-0.24,-0.23,-0.20,1.00,-0.62,0.27,-0.18,0.78,0.42,0.06,0.00,-0.24,0.11,-0.08,0.46,0.98,0.02,-0.03,-0.85,-0.13,-0.17,0.73,0.13,-0.35,-0.88,0.34,-0.40,-0.15,-0.85,0.83,-0.06,0.93,-0.26,0.34,0.09,-0.76,-0.24,0.27,-0.31,0.26,-0.14,0.40,-0.20,0.30,-1.00,0.18,0.06,-0.17,0.09,0.09,0.08,-0.95,0.14,-0.58,0.25,0.43,-0.09,-0.18,-1.00,0.97,0.65,-0.94,-0.06,-0.15,0.22,0.31,-0.61,-0.71,0.08,0.09,0.01,0.48,-0.08,0.60,0.47,-0.68,-0.18,-0.14,-0.31,0.29,-0.22,-1.00,-0.15,0.88,-0.34,0.10,0.38,0.08,-0.16,0.12,-0.15,0.93,-0.33,-0.64,-0.37,1.00,-0.23,0.06,-0.97,-0.79,0.45,-0.54,-0.24,-0.45,-0.01,-0.47,0.98,-0.99,0.14,-0.26,0.01,0.35,0.02,0.06,-0.16,-0.19,-0.76,0.87,-0.00,0.18,-0.24,-0.99,-0.98,-0.76,-0.20,-0.62,-0.37,0.09,0.33,-0.30,0.26,-0.49,0.20,-0.34,0.03,-0.08,-0.14,-0.51,-0.04,0.20,1.00,0.95,-0.45,-0.53,-0.39,-0.20,-0.29,0.99,-0.24,0.97,-0.42,-0.11,0.37,-0.34,-0.37,0.30,0.37,-0.35,0.15,-0.90,-0.98,0.14,-0.21,0.27,-0.54,0.53,-0.21,-0.14,0.98,-0.90,-0.18,-0.35,-0.15,0.19,0.98,0.33,0.12,-0.30,-0.99,-0.27,-0.33,-0.67,0.99,0.16,-0.20,-0.38,1.00,-0.12,-0.94,0.19,-0.04,0.59,-0.10,-0.54,0.22,0.09,-0.19,-0.09,-0.37,-0.14,-0.20,-0.19,-0.67,-0.42,0.16,-0.32,-0.20,-0.46,-0.28,-0.01,-0.22,-0.12,0.32,-0.27,-0.11,0.19,0.25,0.07,0.20,0.31,-0.03,-0.27,0.19,0.22,0.95,0.99,-0.45,0.37,0.04,0.08,0.46,-0.21,-0.29,-0.76,0.88,-0.14,-0.22,-0.14,0.99,0.85,0.46,0.26,0.11,0.55,0.46,0.46,0.39,1.00,-0.07,-0.43,-1.00,-0.44,0.17,-0.05,-0.27,-0.30,0.99,-0.88,-0.34,-0.15,0.46,-0.59,-0.60,-0.22,0.92,-0.06,0.14,-0.24,-0.43,0.19,0.39,-0.35,-0.42,0.40,-0.45,-0.25,-0.09,0.02,-0.24,0.16,0.42,0.15,-0.98,1.00,-0.65,0.43,-0.19,0.10,-0.27,-0.70,0.20,-0.26,-0.94,0.34,0.07,0.25,-0.13,0.08,-0.98,-0.35,-0.54,-0.11,0.01,0.87,0.08,0.08,-0.15,-0.91,0.02,0.15,-0.05,-0.19,0.21,0.97,-0.29,-0.97,-0.14,-0.19,0.13,0.37,0.77,-0.23,-0.51,0.21,0.98,-0.46,-0.38,-0.13,0.28,-0.54,-0.19,-0.32,0.07,-0.40,-0.06,-0.29,-0.21,-0.99,-0.21,0.16,-0.85,-0.00,-0.26,-0.16,-0.85,0.33,-0.10,0.42,0.41,0.09,-0.94,-0.96,0.85,-0.06,-0.42,-0.98,-0.94,-0.39,0.99,-0.43,-1.00,0.25,0.94,0.31,0.28,-0.35,0.30,-0.02,0.31,-0.19,-0.54,-0.07,0.98,0.58,-0.42,0.04,-0.19,1.00,0.17,-0.29,-0.98,-0.15,-0.03,0.29,-0.14,0.29,0.08,0.37,-1.00,-0.29,0.27,0.90,0.55,-0.90,0.05,-0.28,-0.99,0.59,-0.18,0.11,-0.40,-0.26,-0.08,0.93,0.20,-0.30,-0.28,0.35,0.07,0.99,-0.20,0.23,0.28,0.42,1.00,0.45,-0.53,0.99,-0.12,0.34,-0.19,-0.91,0.08,0.84,-0.02,-0.00,0.27,0.64,0.21,-0.16,0.89,1.00,1.00,-0.20,0.34,0.41,-0.16,-0.22,0.25,0.37,0.23,0.88,0.52,0.10,0.16,-0.99,-0.13,0.47,-0.23,-0.17,-0.53,-0.01,0.88,-0.28,-0.03,0.65,-0.21,-0.04,-0.17,-0.15,-0.99,-0.00,0.30,-0.12,0.10,1.00,-0.35,-0.24,0.98,0.04,0.53,-0.23,-0.42,-0.14,-0.04,-0.04,0.97,-0.98,0.21,0.98,-1.00,-0.38,-0.39,0.34,0.34,-0.99,0.20,0.35,0.38,-0.99,0.99,-1.00,0.29,-0.05,0.23,-0.95,0.51,0.04,0.40,0.16,-0.28,0.82,0.98,-0.03,-0.12,-0.15,-0.10,0.04,-0.21,-0.98,-0.67,0.02,0.26,-0.15,0.40,0.61,-0.11,-0.65,-0.42,-0.12,-0.33,0.86,-0.36,-0.87,0.23,0.98,-0.90,-0.18,-0.34,-0.49,0.24,-0.98,0.45,-0.32,0.36,-0.26,-0.24,0.22,0.55,-1.00,0.24,0.18,-0.42,-0.09,0.77,0.10,-0.46,-0.51,-1.00,-0.32,0.30,1.00,0.92,-0.42,0.11,0.24,0.17,-0.86,-0.07,0.23,-0.43,0.25,0.43,-0.28,0.00,0.29,-0.90,0.10,-0.98,-0.99,-0.55,-0.16,0.87,0.19,0.47,-0.94,0.22,-0.06,-0.16,-0.99,-0.34,0.17,-0.20,0.11,0.85,0.91,0.06,-0.44,-0.34,-0.02,0.29,1.00,-0.08,0.43,0.01,0.99,0.17,0.18,0.15,0.21,-0.89,0.95,0.34,0.41,0.34,0.04,-0.25,0.78,-0.43,-0.29,-0.94,0.31,0.53,-0.05,-0.15,0.10,0.25,-0.52],
          "vec_weight": 10.0,
          "bm25_weight": 1.0
        }
      }
    }
  }
}

'''