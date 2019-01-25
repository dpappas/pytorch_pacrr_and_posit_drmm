
import json
from pprint import pprint

f1 = '/home/dpappas/DOC_CNN_PDRMM_standard_feats_run_2/v3 test_gold_bioasq.json'
f2 = '/home/dpappas/DOC_CNN_PDRMM_standard_feats_run_2/v3 test_emit_bioasq.json'

d1 = json.load(open(f1))
d2 = json.load(open(f2))

for tt in zip(d1['questions'], d2['questions']):
    bod = tt[0]['body']
    for doc in list(set(tt[0]['documents'])-set(tt[1]['documents'])):
        print(bod, doc, 'FN')
    for doc in list(set(tt[1]['documents'])-set(tt[0]['documents'])):
        print(bod, doc, 'FP')

'''

ask manolis how many epochs he trained elmo for
ask manolis to train for more epochs
use embeddings that ryan will give me


'''





