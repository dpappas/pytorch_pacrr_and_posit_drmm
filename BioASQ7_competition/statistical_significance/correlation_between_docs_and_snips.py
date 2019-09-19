
import os, json
from pprint import pprint
from scipy.stats import spearmanr
import numpy as np

# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/rerank-batch1-sys1.json'))
# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys1.json'))
# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys2.json'))
# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys3.json'))
# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys4.json'))
data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys5.json'))

# pprint(data)

corrs = []
for q in data['questions']:
    data1 = [int(d.replace('http://www.ncbi.nlm.nih.gov/pubmed/','')) for d in q['documents']]
    data2 = [int(sn['document'].replace('http://www.ncbi.nlm.nih.gov/pubmed/','')) for sn in q['snippets']][:len(data1)]
    print(data1)
    print(data2)
    if(len(data1) == 1):
        corr = 1.0
    else:
        corr, _ = spearmanr(data1, data2)
    corrs.append(corr)

print('Spearmans correlation: %.3f' % np.average(corrs))


