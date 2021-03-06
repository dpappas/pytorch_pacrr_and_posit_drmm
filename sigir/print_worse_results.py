

import json, re
from pprint import pprint
from tqdm import tqdm

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def snip_is_relevant(one_sent, gold_snips):
    one_sent    = ' '.join(bioclean(one_sent)).strip()
    gold_snips  = [' '.join(bioclean(snip)).strip() for snip in gold_snips]
    return int(one_sent in gold_snips)

def get_one_bad():
    for snip in val['snippets'][pmid]:
        gold_truth = snip[0]
        emit = snip[1]
        sent = snip[-1]
        if(emit-gold_truth>.70):
            return {
                    'question_text': qtext,
                    'pmid': pmid,
                    'gold_truth': gold_truth,
                    'emit': emit,
                    'sent': sent,
                    'gold_snip': 'NONE FOUND'
                }
    return None

def get_one_good():
    for snip in val['snippets'][pmid]:
        gold_truth  = snip[0]
        emit        = snip[1]
        sent        = snip[-1]
        if(gold_truth-emit>.90):
            for gold_snip in gdata[qid]:
                if(sent in gold_snip or gold_snip in sent):
                    return(
                        {
                            'question_text' : qtext,
                            'pmid'          : pmid,
                            'gold_truth'    : gold_truth,
                            'emit'          : emit,
                            'sent'          : sent,
                            'gold_snip'     : gold_snip
                        }
                    )
    return None

def get_all_bad():
    ret = []
    for snip in val['snippets'][pmid]:
        gold_truth = snip[0]
        emit = snip[1]
        sent = snip[-1]
        if(emit-gold_truth>.70):
            ret.append(
                {
                    'question_text': qtext,
                    'pmid': pmid,
                    'gold_truth': gold_truth,
                    'emit': emit,
                    'sent': sent,
                    'gold_snip': 'NONE FOUND'
                }
            )
    return ret

def get_all_good():
    ret = []
    for snip in val['snippets'][pmid]:
        gold_truth  = snip[0]
        emit        = snip[1]
        sent        = snip[-1]
        if(gold_truth-emit>.80):
            for gold_snip in gdata[qid]:
                if(sent in gold_snip or gold_snip in sent):
                    ret.append(
                        {
                            'question_text' : qtext,
                            'pmid'          : pmid,
                            'gold_truth'    : gold_truth,
                            'emit'          : emit,
                            'sent'          : sent,
                            'gold_snip'     : gold_snip
                        }
                    )
    return ret

import pickle
dataloc = '/home/dpappas/for_ryan/'
with open(dataloc + 'bioasq_bm25_top100.dev.pkl', 'rb') as f:
    dev_data = pickle.load(f)

# pprint(
#     [
#         d for d in dev_data['queries']
#         if(d['query_id'] == '58a93877ee23e0236b000001')
#     ]
# )
# exit()

fpath = '/home/dpappas/v3 dev_data_for_revision.json'
gpath = '/home/dpappas/v3 dev_gold_bioasq.json'
opath = '/home/dpappas/some_data_for_revision.json'

edata   = json.load(open(fpath))

data    = json.load(open(gpath))
gdata   = {}
for item in tqdm(data['questions']):
    all_snips           = [sn['text'] for sn in item['snippets']]
    gdata[item['id']]   = all_snips

all_out_data_good = []
all_out_data_bad = []
for qid, val in tqdm(edata.items()):
    qtext = val['query_text']
    for pmid in val['snippets']:
        found_good  = False
        found_bad   = False
        b           = get_one_bad()
        if(b is not None):
            all_out_data_bad.append(b)
        g           = get_one_good()
        if(g is not None):
            all_out_data_good.append(g)

with open(opath, 'w') as f:
    all_out_data = {
        'GT_1': all_out_data_good,
        'GT_0': all_out_data_bad
    }
    f.write(json.dumps(all_out_data, indent=4, sort_keys=True))
    f.close()

############################

all_out_data_good   = []
all_out_data_bad    = []
for qid, val in tqdm(edata.items()):
    qtext = val['query_text']
    for pmid in val['snippets']:
        all_out_data_bad.extend(get_all_bad())
        all_out_data_good.extend(get_all_good())

opath = '/home/dpappas/all_data_for_revision.json'
with open(opath, 'w') as f:
    all_out_data = {
        'GT_1': sorted(all_out_data_good, key=lambda x: x['emit'], reverse=False),
        'GT_0': sorted(all_out_data_bad, key=lambda x: x['emit'], reverse=True)
    }
    f.write(json.dumps(all_out_data, indent=4, sort_keys=True))
    f.close()

'''

import torch 
# data                        = torch.randn(3, 40)
data                        = torch.tensor([
[1, 2, 20, 3, 40, 5, 4, 10, 0],
[1, 2, 3, 4, 5, 6, 7, 8, 9],
[1, 2, 20, 3, 40, 5, 4, 10, 11]
])
def get_distance_feature(data):
    sorted_data, sorted_indices = torch.sort(data, -1)
    print(sorted_data[:,-5:])
    print(sorted_indices[:,-5:])
    maxk_inds                   = sorted_indices[:, -5:].float()
    maxk_inds                   /= sorted_indices.max(dim=-1)[0].float().unsqueeze(-1).expand_as(maxk_inds) 
    sorted_maxk_inds            = torch.sort(maxk_inds, -1)[0]
    print(sorted_maxk_inds)
    dist_feat = None
    for i in range(sorted_maxk_inds.size(-1)-1):
        temp        = sorted_maxk_inds[:,i+1] - sorted_maxk_inds[:,i] 
        dist_feat   = (dist_feat+temp) if(dist_feat is not None) else temp 
    return dist_feat

print(get_distance_feature(data))

'''
