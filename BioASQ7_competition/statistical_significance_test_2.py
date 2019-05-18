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

# def ranking_rprecision_score(y_true, y_score, k=10):
#     """Precision at rank k
#     Parameters
#     ----------
#     y_true : array-like, shape = [n_samples]
#         Ground truth (true relevance labels).
#     y_score : array-like, shape = [n_samples]
#         Predicted scores.
#     k : int
#         Rank.
#     Returns
#     -------
#     precision @k : float
#     """
#     unique_y = np.unique(y_true)
#
#     if len(unique_y) == 1:
#         return ValueError("The score cannot be approximated.")
#     elif len(unique_y) > 2:
#         raise ValueError("Only supported for two relevance levels.")
#
#     pos_label = unique_y[1]
#     n_pos = np.sum(y_true == pos_label)
#
#     order = np.argsort(y_score)[::-1]
#     y_true = np.take(y_true, order[:k])
#     n_relevant = np.sum(y_true == pos_label)
#
#     # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
#     return float(n_relevant) / min(k, n_pos)
#
# def mean_rprecision_k(y_true, y_score, k=10):
#     """Mean precision at rank k
#     Parameters
#     ----------
#     y_true : array-like, shape = [n_samples]
#         Ground truth (true relevance labels).
#     y_score : array-like, shape = [n_samples]
#         Predicted scores.
#     k : int
#         Rank.
#     Returns
#     -------
#     mean precision @k : float
#     """
#
#     p_ks = []
#     for y_t, y_s in zip(y_true, y_score):
#         if np.sum(y_t == 1):
#             p_ks.append(ranking_rprecision_score(y_t, y_s, k=k))
#
#     return np.mean(p_ks)
#
# def get_score(gold, predictions, metric='r-precision'):
#     if metric == 'r-precision':
#         return mean_rprecision_k(gold,predictions, k=5)
#     else:
#         pred_targets = (predictions > 0.5).astype('int32')
#         return f1_score(y_true=gold, y_pred=pred_targets, average='micro')

def get_bioasq_res(fgold, femit):
    '''
    java -Xmx10G -cp /home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar
    evaluation.EvaluatorTask1b -phaseA -e 5
    /home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/BioASQ-task6bPhaseB-testset1
    ./drmm-experimental_submit.json
    '''
    jar_path = retrieval_jar_path
    #
    bioasq_eval_res = subprocess.Popen(
        ['java', '-Xmx10G', '-cp', jar_path, 'evaluation.EvaluatorTask1b', '-phaseA', '-e', '5', fgold, femit],
        stdout=subprocess.PIPE, shell=False
    )
    (out, err) = bioasq_eval_res.communicate()
    lines = out.decode("utf-8").split('\n')
    ret = {}
    for line in lines:
        if (':' in line):
            k = line.split(':')[0].strip()
            v = line.split(':')[1].strip()
            ret[k] = float(v)
    return ret

retrieval_jar_path  = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'
odir                = '/home/dpappas/sign_testing/'

goldf = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1'

# sysAf = '/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/jpdrmm.json'
# sysBf = '/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/pdrmm.json'
# sysAf = '/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/bert.json'
# sysBf = '/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/JBERT.json'
sysAf = '/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/JBERT_F.json'
sysBf = '/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/JBERT.json'
temp1f = 'A1.json'
temp2f = 'A2.json'

print(goldf)
print(sysAf)
print(sysBf)

scoreA = get_bioasq_res(goldf, sysAf)[u'MAP documents']
scoreB = get_bioasq_res(goldf, sysBf)[u'MAP documents']
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

for n in tqdm.tqdm(range(1, N+1)):
    A = copy.deepcopy(sysA)
    B = copy.deepcopy(sysB)
    ########################
    for j in range(len(A['questions'])):
        for i in range(len(A['questions'][j]['documents'])):
            rval = random.random()
            if rval < 0.5:
                temp = A['questions'][j]['documents'][i]
                A['questions'][j]['documents'][i] = B['questions'][j]['documents'][i]
                B['questions'][j]['documents'][i] = temp
    with open(temp1f, 'w') as f:
        json.dump(A, f)
    with open(temp2f, 'w') as f:
        json.dump(B, f)
    new_sysA_metric = get_bioasq_res(goldf, temp1f)[u'MAP documents']
    new_sysB_metric = get_bioasq_res(goldf, temp2f)[u'MAP documents']
    new_diff        = abs(new_sysA_metric - new_sysB_metric)
    if new_diff >= orig_diff:
        num_invalid += 1
    if n % 20 == 0 and n > 0:
        print('Random Iteration {}: {}'.format(n, float(num_invalid) / float(n)))

print('Overall: {}'.format(float(num_invalid) / float(N)))
