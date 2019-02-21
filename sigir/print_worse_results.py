

import json
from pprint import pprint
from tqdm import tqdm

fpath = '/home/dpappas/v3 dev_data_for_revision.json'
gpath = '/home/dpappas/v3 dev_gold_bioasq.json'

edata = json.load(open(fpath))

data    = json.load(open(gpath))
gdata  = {}
for item in tqdm(data['questions']):
    all_snips           = [sn['text'] for sn in item['snippets']]
    gdata[item['id']]   = all_snips

for qid, val in tqdm(edata.items()):
    qtext = val['query_text']
    for pmid in val['snippets']:
        for snip in val['snippets'][pmid]:
            g = snip[0]
            e = snip[1]
            sent = snip[-1]
            if(g-e>.90):
                snips_containing    = [sn for sn in gdata[qid] if(sn in sent or sent in sn)]
                sn                  = snips_containing[0] if(len(snips_containing)>0) else 'NONE'
                print('{} | {} | {} | {} | {} | {}'.format(qtext, pmid, g, e, sent, sn))
                # print(20 * '-')





















