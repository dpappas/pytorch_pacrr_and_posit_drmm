
from tqdm           import tqdm
from nltk.tokenize  import sent_tokenize, word_tokenize
from elasticsearch  import Elasticsearch
from elasticsearch.helpers import bulk, scan
from collections    import Counter
import pickle, re

bioclean_mine = lambda t: re.sub(u'[.,?;*!%^&_+():-\[\]{}…"\'§£/ˈ‡†ʔ✚=°→€\\\\]', '', t.strip().lower()).split()

################################################
doc_index       = 'natural_questions_0_1'
doc_map         = "natural_questions_map_0_1"
################################################

es      = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
items   = scan(es, query=None, index=doc_index, doc_type=doc_map)
################################################

df = Counter()
df2 = Counter()
total = es.count(index=doc_index)['count']
for item in tqdm(items, total=total):
    df.update(Counter(list(set([t.lower() for t in word_tokenize(item['_source']['paragraph_text'])]))))
    df2.update(Counter(list(set([t.lower() for t in bioclean_mine(item['_source']['paragraph_text'])]))))


pickle.dump(df, open('NQ_bioclean.pkl', 'wb'))
pickle.dump(df2, open('NQ_bioclean_mine_df.pkl', 'wb'))

'''
import pickle
from pprint import pprint

d = pickle.load(open('NQ_df.pkl','rb'))
d = sorted(d.items(), key=lambda x: x[1], reverse=True)
pprint(d[-20:])

d2 = pickle.load(open('NQ_bioclean_mine_df.pkl','rb'))
d2 = sorted(d2.items(), key=lambda x: x[1], reverse=True)
pprint(d2[-20:])

print(len(d), len(d2))

'''


