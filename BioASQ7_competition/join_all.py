import pickle
import random
import numpy as np
import tqdm
from sklearn.metrics import f1_score
from pprint import pprint
import os
import json
import copy
import subprocess
import sys

names = [
'bert_bcnn.json', 'bertHC_pdrmm.json', 'bert_high_bcnn.json', 'bert_pdrmm.json', 'pdrmm_bcnn.json',
'pdrmm_pdrmm.json', 'term_pacrr_bcnn.json'
]

odir = '/home/dpappas/bioasq_all/bioasq7/snippet_results/b123_joined/'
os.makedirs(odir)
all_data = {'questions':[]}

for name in names:
    for b in range(1, 4):
        data = json.load(open('/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_{}/{}'.format(b, name)))['questions']
        all_data['questions'].extend(data)
    with open(os.path.join(odir, name), 'w') as f:
        f.write(json.dumps(all_data, indent=4, sort_keys=True))
        f.close()

##################################################################################################

names = [
'bert-high-conf-0.01.json', 'bert_jpdrmm.json', 'bert.json', 'JBERT_F.json', 'JBERT.json',
'jpdrmm.json', 'pdrmm.json', 'term-pacrr.json'
]

odir = '/home/dpappas/bioasq_all/bioasq7/document_results/b123_joined/'
os.makedirs(odir)
all_data = {'questions':[]}

for name in names:
    for b in range(1, 4):
        data = json.load(open('/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_{}/{}'.format(b, name)))['questions']
        all_data['questions'].extend(data)
    with open(os.path.join(odir, name), 'w') as f:
        f.write(json.dumps(all_data, indent=4, sort_keys=True))
        f.close()

##################################################################################################

for b in range(1, 4):
    data = json.load(
        open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(b, b))
    )['questions']
    all_data['questions'].extend(data)

with open(os.path.join('/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset123'), 'w') as f:
    f.write(json.dumps(all_data, indent=4, sort_keys=True))
    f.close()

##################################################################################################

# /home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1


goldf, sysAf, sysBf, metric, opath = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

rand_name   = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz'+'abcdefghijklmnopqrstuvwxyz'.upper()) for i in range(4)])
temp1f      = '{}.json'.format(rand_name+'A')
temp2f      = '{}.json'.format(rand_name+'B')

print(goldf)
print(sysAf)
print(sysBf)

scoreA  = get_bioasq_res(goldf, sysAf)[metric]
scoreB  = get_bioasq_res(goldf, sysBf)[metric]
pprint(scoreA)
pprint(scoreB)

sysA    = json.load(open(sysAf))
sysB    = json.load(open(sysBf))
# pprint(sysA['questions'][0])
# pprint(sysΒ['questions'][0])

sysA_metric = scoreA
sysB_metric = scoreB
orig_diff   =  abs(sysA_metric - sysB_metric)

N           = 10000
num_invalid = 0

pbar = tqdm.tqdm(range(1, N+1))
for n in pbar:
    A = copy.deepcopy(sysA)
    B = copy.deepcopy(sysB)
    ########################
    for qi in range(len(A['questions'])):
        rval = random.random()
        if rval < 0.5:
            if(metric=="MAP snippets"):
                AD = [d for d in A['questions'][qi]["snippets"]]
                BD = [d for d in B['questions'][qi]["snippets"]]
                A['questions'][qi]["snippets"] = [d for d in BD]
                B['questions'][qi]["snippets"] = [d for d in AD]
            else:
                AD = [d for d in A['questions'][qi]['documents']]
                BD = [d for d in B['questions'][qi]['documents']]
                A['questions'][qi]['documents'] = [d for d in BD]
                B['questions'][qi]['documents'] = [d for d in AD]
    ########################
    with open(temp1f, 'w') as f:
        json.dump(A, f)
    with open(temp2f, 'w') as f:
        json.dump(B, f)
    new_sysA_metric = get_bioasq_res(goldf, temp1f)[metric]
    new_sysB_metric = get_bioasq_res(goldf, temp2f)[metric]
    new_diff        = abs(new_sysA_metric - new_sysB_metric)
    if new_diff >= orig_diff:
        num_invalid += 1
    if n % 20 == 0 and n > 0:
        pbar.set_description('Random Iteration {}: {}'.format(n, float(num_invalid) / float(n)))

print('Overall: {}'.format(float(num_invalid) / float(N)))
with(open(opath, 'w')) as fp:
    fp.write('Overall: {}'.format(float(num_invalid) / float(N)))
    fp.close()

