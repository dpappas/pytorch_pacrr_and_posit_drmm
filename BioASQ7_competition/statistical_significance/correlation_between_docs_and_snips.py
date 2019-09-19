
import sys
from scipy import stats
import os, json
from pprint import pprint
from scipy.stats import spearmanr
import numpy as np

#Permutation-randomization
#Repeat R times: randomly flip each m_i(A),m_i(B) between A and B with probability 0.5, calculate delta(A,B).
# let r be the number of times that delta(A,B)<orig_delta(A,B)
# significance level: (r+1)/(R+1)
# Assume that larger value (metric) is better
def rand_permutation(data_A, data_B, n, R):
    delta_orig = float(sum([ x - y for x, y in zip(data_A, data_B)]))/n
    r = 0
    for x in range(0, R):
        temp_A = data_A
        temp_B = data_B
        # samples = [np.random.randint(1, 3) for i in xrange(n)] #which samples to swap without repetitions
        samples = [np.random.randint(1, 3) for i in range(n)] #which samples to swap without repetitions
        swap_ind = [i for i, val in enumerate(samples) if val == 1]
        for ind in swap_ind:
            temp_B[ind], temp_A[ind] = temp_A[ind], temp_B[ind]
        delta = float(sum([ x - y for x, y in zip(temp_A, temp_B)]))/n
        if(delta<=delta_orig):
            r = r+1
    pval = float(r+1.0)/(R+1.0)
    return pval

#Bootstrap
#Repeat R times: randomly create new samples from the data with repetitions, calculate delta(A,B).
# let r be the number of times that delta(A,B)<2*orig_delta(A,B). significance level: r/R
# This implementation follows the description in Berg-Kirkpatrick et al. (2012),
# "An Empirical Investigation of Statistical Significance in NLP".
def Bootstrap(data_A, data_B, n, R):
    delta_orig = float(sum([x - y for x, y in zip(data_A, data_B)])) / n
    r = 0
    for x in range(0, R):
        temp_A = []
        temp_B = []
        samples = np.random.randint(0,n,n) #which samples to add to the subsample with repetitions
        for samp in samples:
            temp_A.append(data_A[samp])
            temp_B.append(data_B[samp])
        delta = float(sum([x - y for x, y in zip(temp_A, temp_B)])) / n
        if (delta < 2*delta_orig):
            r = r + 1
    pval = float(r)/(R)
    return pval

## McNemar test
def calculateContingency(data_A, data_B, n):
    ABrr = 0
    ABrw = 0
    ABwr = 0
    ABww = 0
    for i in range(0,n):
        if(data_A[i]==1 and data_B[i]==1):
            ABrr = ABrr+1
        if (data_A[i] == 1 and data_B[i] == 0):
            ABrw = ABrw + 1
        if (data_A[i] == 0 and data_B[i] == 1):
            ABwr = ABwr + 1
        else:
            ABww = ABww + 1
    return np.array([[ABrr, ABrw], [ABwr, ABww]])

def mcNemar(table):
    statistic = float(np.abs(table[0][1]-table[1][0]))**2/(table[1][0]+table[0][1])
    pval = 1-stats.chi2.cdf(statistic,1)
    return pval

def do_the_test(data_A, data_B, alpha):
    data_A = list(map(float, data_A))
    data_B = list(map(float, data_B))
    name = "Permutation"
    ### Statistical tests
    # Paired Student's t-test: Calculate the T-test on TWO RELATED samples of scores, a and b. for one sided test we multiply p-value by half
    if(name=="t-test"):
        t_results = stats.ttest_rel(data_A, data_B)
        # correct for one sided test
        pval = float(t_results[1]) / 2
        if (float(pval) <= float(alpha)):
            print("\nTest result is significant with p-value: {}".format(pval))
            return
        else:
            print("\nTest result is not significant with p-value: {}".format(pval))
            return
    # Wilcoxon: Calculate the Wilcoxon signed-rank test.
    if(name=="Wilcoxon"):
        wilcoxon_results = stats.wilcoxon(data_A, data_B)
        if (float(wilcoxon_results[1]) <= float(alpha)):
            print("\nTest result is significant with p-value: {}".format(wilcoxon_results[1]))
            return
        else:
            print("\nTest result is not significant with p-value: {}".format(wilcoxon_results[1]))
            return
    #
    if(name=="McNemar"):
        print("\nThis test requires the results to be binary : A[1, 0, 0, 1, ...], B[1, 0, 1, 1, ...] for success or failure on the i-th example.")
        f_obs = calculateContingency(data_A, data_B, len(data_A))
        mcnemar_results = mcNemar(f_obs)
        if (float(mcnemar_results) <= float(alpha)):
            print("\nTest result is significant with p-value: {}".format(mcnemar_results))
            return
        else:
            print("\nTest result is not significant with p-value: {}".format(mcnemar_results))
            return
    #
    if(name=="Permutation"):
        R = max(10000, int(len(data_A) * (1 / float(alpha))))
        pval = rand_permutation(data_A, data_B, len(data_A), R)
        print('pval : {}'.format(pval))
        if (float(pval) <= float(alpha)):
            print("\nTest result is significant with p-value: {}".format(pval))
            return float(pval)
        else:
            print("\nTest result is not significant with p-value: {}".format(pval))
            return float(pval)
    #
    if(name=="Bootstrap"):
        R = max(10000, int(len(data_A) * (1 / float(alpha))))
        pval = Bootstrap(data_A, data_B, len(data_A), R)
        if (float(pval) <= float(alpha)):
            print("\nTest result is significant with p-value: {}".format(pval))
            return
        else:
            print("\nTest result is not significant with p-value: {}".format(pval))
            return
    #
    else:
        print("\nInvalid name of statistical test")
        sys.exit(1)

# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/rerank-batch1-sys1.json'))
# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys1.json'))
# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys2.json'))
# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys3.json'))
# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys4.json'))
# data = json.load(open('/home/dpappas/bioasq_all/bioasq7/submit_files/test_batch_1/batch1-sys5.json'))

# data = json.load(open('/home/dpappas/test_pdrmm_pdrmm_batch4/v3 test_emit_bioasq.json'))
# data = json.load(open('/home/dpappas/test_jpdrmm_high_batch3/v3 test_emit_bioasq.json'))
# data = json.load(open('/media/dpappas/dpappas_data/models_out/test_jpdrmm_high_batch1/v3 test_emit_bioasq.json'))
data = json.load(open('/media/dpappas/dpappas_data/models_out/test_bert_jpdrmm_high_batch3/v3 test_emit_bioasq.json'))

# data = json.load(open('/media/dpappas/dpappas_data/models_out/test_snippet_bert_pdrmm_batch1/v3 test_emit_bioasq.json'))

# pprint(data)

p_vals = []
for q in data['questions']:
    data1 = [int(d.replace('http://www.ncbi.nlm.nih.gov/pubmed/','')) for d in q['documents']]
    data2 = [int(sn['document'].replace('http://www.ncbi.nlm.nih.gov/pubmed/','')) for sn in q['snippets']][:len(data1)]
    # print(data1)
    # print(data2)
    p_val = do_the_test(data1, data2, 0.05)
    p_vals.append(p_val)

print(np.average(p_vals))

'''
corrs = []
for q in data['questions']:
    data1 = [int(d.replace('http://www.ncbi.nlm.nih.gov/pubmed/','')) for d in q['documents']]
    data2 = [int(sn['document'].replace('http://www.ncbi.nlm.nih.gov/pubmed/','')) for sn in q['snippets']][:len(data1)]
    # print(data1)
    # print(data2)
    if(len(data1) == 1):
        corr = 1.0
    else:
        corr, _ = spearmanr(data1, data2)
    corrs.append(corr)

print('Spearmans correlation: %.3f' % np.average(corrs))
'''

