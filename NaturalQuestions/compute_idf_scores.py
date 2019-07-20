
from tqdm           import tqdm
from nltk.tokenize  import sent_tokenize, word_tokenize
from elasticsearch  import Elasticsearch
from elasticsearch.helpers import bulk, scan
from collections    import Counter
import pickle

################################################
doc_index       = 'natural_questions_0_1'
doc_map         = "natural_questions_map_0_1"
################################################

es      = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
items   = scan(es, query=None, index=doc_index, doc_type=doc_map)
################################################

df = Counter()
total = es.count(index=doc_index)['count']
for item in tqdm(items, total=total):
    df.update(Counter(list(set([t.lower() for t in word_tokenize(item['_source']['paragraph_text'])]))))

pickle.dump(df, open('NQ_df.pkl', 'wb'))

'''
import pickle
from pprint import pprint
d = pickle.load(open('NQ_df.pkl','rb'))
d = sorted(d.items(), key=lambda x: x[1], reverse=True)
pprint(d[:20])
'''



