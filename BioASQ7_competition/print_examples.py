


import json, re
import difflib
from pprint import pprint
from pprint import pprint

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

batch = 3
d   = json.load(open('/home/dpappas/test_jpdrmm_high_batch{}/v3 test_data_for_revision.json'.format(batch)))
d1   = json.load(open('/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_{}/jpdrmm.json'.format(batch)))
d2  = json.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(batch, batch)))

data1 = {}
for q in d1['questions']:
    tt = {}
    for sn in q['snippets']:
        if(sn['document'] in tt):
            tt[sn['document']].append(sn['text'])
        else:
            tt[sn['document']] = [sn['text']]
    data1[q['body']] = tt

data2 = {}
for q in d2['questions']:
    pprint(q)
    tt = {}
    for sn in q['snippets']:
        if(sn['document'] in tt):
            tt[sn['document']].append(sn['text'])
        else:
            tt[sn['document']] = [sn['text']]
    data2[q['body']] = tt

'''
all_data = []
for v in d.values():
    for k in v['snippets']:
        for sn in v['snippets'][k]:
            # if(sn[0]==1 and sn[1]>0.5 and len(sn[-1])<150):
            if(sn[1] > 0.5 and len(sn[-1]) < 150):
                t1 = set(bioclean(v['query_text']))
                t2 = set(bioclean(sn[-1]))
                dif = t1 - t2
                if(len(dif)/float(len(t1))>0.2):
                    all_data.append(sn + [v['query_text']])

all_data = sorted(all_data, key=lambda x: x[1], reverse=True)

pprint(all_data)
'''

pprint(data1[u'What is CardioClassifier?'])
pprint(data2[u'What is CardioClassifier?'])

for q in data1:
    for d in data1[q]:
        if(d in data2[q]):
            if(len(data1[q][d]) - len(data2[q][d]) == 0):
                print(d)
                print(q)
                pprint(data1[q][d])
                print(5 * '#')
                pprint(data2[q][d])
                print(40 * '-')

pprint(data1[u'Please list the 4 genes involved in Sanfilippo syndrome, also known as mucopolysaccharidosis III (MPS-III).'])
pprint(data2[u'Please list the 4 genes involved in Sanfilippo syndrome, also known as mucopolysaccharidosis III (MPS-III).'])

























