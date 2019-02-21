

import json
from pprint import pprint
from tqdm import tqdm

def snip_is_relevant(one_sent, gold_snips):
    # print one_sent
    # pprint(gold_snips)
    return int(
        any(
            [
                (one_sent.encode('ascii', 'ignore')  in gold_snip.encode('ascii','ignore'))
                or
                (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii','ignore'))
                for gold_snip in gold_snips
            ]
        )
    )

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
            # pprint({
            #         'question_text': qtext,
            #         'pmid': pmid,
            #         'gold_truth': gold_truth,
            #         'emit': emit,
            #         'sent': sent,
            #         'gold_snip': 'NONE FOUND'
            #     })
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

fpath = '/home/dpappas/v3 dev_data_for_revision.json'
gpath = '/home/dpappas/v3 dev_gold_bioasq.json'
opath = '/home/dpappas/some_data_for_revision.json'

edata = json.load(open(fpath))

data    = json.load(open(gpath))
gdata  = {}
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

# exit()

with open(opath, 'w') as f:
    all_out_data = {
        'GT_1': all_out_data_good,
        'GT_0': all_out_data_bad
    }
    f.write(json.dumps(all_out_data, indent=4, sort_keys=True))
    f.close()


