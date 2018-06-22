

from bioasq_utils import get_sents
from pprint import pprint
import json

fpath   = '/home/dpappas/bioasq_ir_data/BioASQ-trainingDataset6b.json'
data    = json.load(open(fpath,'r'))

# pprint(data)

pprint(data['questions'][0].keys())
pprint(data['questions'][0])

bod     = data['questions'][0]['body']
print bod
doc_ids = [ t.split('/')[-1] for t in data['questions'][0]['documents'] ]
print doc_ids
















