
from nltk.corpus import stopwords
from bioasq_utils import get_sents
from pprint import pprint
import json

fpath   = '/home/dpappas/bioasq_ir_data/BioASQ-trainingDataset6b.json'
data    = json.load(open(fpath,'r'))

# pprint(data)
sw = set(stopwords.words('english'))

for snip in data['questions'][0]['snippets']:
    # pprint(snip)
    pmid = snip['document'].split('/')[-1]
    print(pmid)
    print(snip['beginSection'])
    print(snip['text'].strip())
    # print(snip['offsetInBeginSection'])
    # print(snip['offsetInEndSection'])
    print 30 * '-'


'''

# pprint(data['questions'][0].keys())
# pprint(data['questions'][0])
#
# bod     = data['questions'][0]['body']
# print bod
# doc_ids = [ t.split('/')[-1] for t in data['questions'][0]['documents'] ]
# print doc_ids


laptop:
/home/dpappas/ELK/elasticsearch-6.2.4/bin/elasticsearch
/home/dpappas/ELK/kibana-6.2.4-linux-x86_64/bin/kibana
server:

'''











