

import json
from pprint import pprint
from tqdm import tqdm

fpath = '/home/dpappas/v3 dev_data_for_revision.json'
gpath = '/home/dpappas/v3 dev_gold_bioasq.json'
opath = '/home/dpappas/some_data_for_revision.json'

edata = json.load(open(fpath))

data    = json.load(open(gpath))
gdata  = {}
for item in tqdm(data['questions']):
    all_snips           = [sn['text'] for sn in item['snippets']]
    gdata[item['id']]   = all_snips

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

all_out_data = []
for qid, val in tqdm(edata.items()):
    qtext = val['query_text']
    for pmid in val['snippets']:
        for snip in val['snippets'][pmid]:
            gold_truth = snip[0]
            emit = snip[1]
            sent = snip[-1]
            if(gold_truth-emit>.90):
                for gold_snip in gdata[qid]:
                    if(sent in snip or snip in sent):
                        all_out_data.append(
                            {
                                'question_text' : qtext,
                                'pmid'          : pmid,
                                'gold_truth'    : gold_truth,
                                'emit'          : emit,
                                'sent'          : sent,
                                'gold_snip'     : gold_snip
                            }
                        )
            elif(emit-gold_truth>.80):
                all_out_data.append(
                    {
                        'question_text': qtext,
                        'pmid': pmid,
                        'gold_truth': gold_truth,
                        'emit': emit,
                        'sent': sent,
                        'gold_snip': 'NONE FOUND'
                    }
                )

with open(opath, 'w') as f:
    f.write(json.dumps(all_out_data, indent=4, sort_keys=True))
    f.close()




















