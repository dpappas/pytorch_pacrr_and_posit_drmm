

import  json
from    pprint import pprint

gold_f = "/home/dpappas/test_bert_jpdrmm/v3 test_gold_bioasq.json"
emit_f = "/home/dpappas/test_bert_jpdrmm/v3 test_emit_bioasq.json"

all_gd  = json.load(open(gold_f))
all_gd  = dict((q['id'], q) for q in all_gd['questions'])
all_ed  = json.load(open(emit_f))
all_ed  = dict((q['id'], q) for q in all_ed['questions'])

for id in all_gd:
    gd      = all_gd[id]
    qtext   = gd['body']
    ed      = all_ed[id]
    s1      = set(gd['documents'])
    s2      = set(ed['documents'])
    dif     = s2.difference(s1)
    for doc_link in dif:
        print(qtext)
        print(doc_link)
        pprint([ sn['text'] for sn in ed['snippets'] if(sn['document'] == doc_link)])
        print(20*'-')
    print(20*'=')
