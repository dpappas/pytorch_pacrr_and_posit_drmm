

import json
import numpy as np

# bert-high-conf-0.01.json  bert_jpdrmm.json  bert.json  JBERT_F.json  JBERT.json  jpdrmm.json  pdrmm.json  term-pacrr.json

fpath1  = "/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/pdrmm.json"
fpath2  = "/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/bert.json"

d1      = json.load(open(fpath1))
d1      = dict((q['id'], q) for q in d1['questions'])
d2      = json.load(open(fpath2))
d2      = dict((q['id'], q) for q in d2['questions'])

common  = []
for k in d1:
    docs1   = d1[k]['documents']
    docs2   = d2[k]['documents']
    comm    = len(set(docs1).intersection(set(docs2)))
    common.append(float(comm) / float(10.))

print(np.average(common))



