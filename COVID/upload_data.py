
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
zip_path    = '/media/dpappas/dpappas_data/COVID/db_entries.zip'
archive     = zipfile.ZipFile(zip_path, 'r')
d           = json.loads(archive.read('db_entries.json'))

index       = 'covid_index_0_1'
elastic_con = Elasticsearch(['127.0.01:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

fromm       = int(sys.argv[1])
too         = int(sys.argv[2])
actions     = []
b_size      = 10
for item in tqdm(d[fromm:too]):
    if(len(item['date'])==0):
        item['date'] = None
    else:
        try:
            item['date'] = parser.parse(item['date']).strftime('%Y-%m-%d')
        except:
            item['date'] = None
    hash_object     = hashlib.md5(item['joint_text'].encode()).hexdigest()
    idd                     = '{}_{}_{}_{}'.format(item['pmid'], item['pmcid'], item['doi'], hash_object)
    vec                     = encode_sent_with_bert(item['joint_text'].replace('\n------------------------------', ''), max_len=512)
    vec                     = vec[0].cpu().detach().numpy()
    item["doc_vec_scibert"] = vec.tolist()
    actions.append(create_an_action(item))
    if (len(actions) >= b_size):
        send_to_elk(actions)
        actions = []

send_to_elk(actions)

'''
python3.6 index_sents.py 0 250000 &
python3.6 index_sents.py 250000 500000 &
python3.6 index_sents.py 500000 750000 &
python3.6 index_sents.py 750000 1000000 &
python3.6 index_sents.py 1000000 1250000
'''