

import json
import numpy as np

# bert-high-conf-0.01.json  bert_jpdrmm.json  bert.json  JBERT_F.json  JBERT.json  jpdrmm.json  pdrmm.json  term-pacrr.json

fpath1  = "/home/dpappas/bioasq_all/bioasq7/document_results/b12345_joined/jpdrmm.json"
fpath2  = "/home/dpappas/bioasq_all/bioasq7/document_results/b12345_joined/JBERT.json"

goldf   = '/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset12345'

d1      = json.load(open(fpath1))
d1      = dict((q['id'], q) for q in d1['questions'])
d2      = json.load(open(fpath2))
d2      = dict((q['id'], q) for q in d2['questions'])
gold_d  = json.load(open(goldf))
gold_d  = dict((q['id'], q) for q in gold_d['questions'])

common          = []
gold_common     = []
gold_not_common = []
for k in d1:
    ######################################################
    docs1       = d1[k]['documents']
    docs2       = d2[k]['documents']
    docsg       = gold_d[k]['documents']
    ######################################################
    koina       = set(docs1).intersection(set(docs2))
    gold_koina  = set(koina).intersection(set(docsg))
    ######################################################
    gold_common.append(float(len(gold_koina)) / float(len(koina)))
    common.append(float(len(koina)) / float(10.))
    ######################################################
    # break

print(np.average(gold_common))
print(np.average(common))

# jpdrmm jbert-NOT-frozen           0.7326
# jpdrmm jbert-frozen               0.7316
# jpdrmm bert                       0.6716
# jbert-frozen   jbert-NOT-frozen   0.9146
# jbert-frozen      bert            0.6598
# jbert-NOT-frozen  bert            0.6600


