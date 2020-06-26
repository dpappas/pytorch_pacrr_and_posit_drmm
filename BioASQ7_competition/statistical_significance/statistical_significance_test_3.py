import os
import json
import sys
import uuid
import datetime
import subprocess
import re
import random

def format_bioasq2treceval_qrels(bioasq_data, filename):
  with open(filename, 'w') as f:
    for q in bioasq_data['questions']:
      for d in q['documents']:
        f.write('{0} 0 {1} 1'.format(q['id'], d))
        f.write('\n')

def format_bioasq2treceval_qret(bioasq_data, system_name, filename):
  with open(filename, 'w') as f:
    for q in bioasq_data['questions']:
      rank = 1
      for d in q['documents']:
        sim = (len(q['documents']) + 1 - rank) / float(len(q['documents'])) # Just a fake similarity. Does not affect the metrics we are using.
        f.write('{0} {1} {2} {3} {4} {5}'.format(q['id'], 0, d, rank, sim, system_name))
        f.write('\n')
        rank += 1

def trec_evaluate(qrels_file, qret_file):
  VERSION = '9.0'
  if VERSION == '8.1':
    trec_eval_res = subprocess.Popen(
        [os.path.dirname(os.path.realpath(__file__)) + '/trec_eval.8.1/./trec_eval', '-a', qrels_file, qret_file],
        stdout=subprocess.PIPE, shell=False)
  elif VERSION == '9.0':
    trec_eval_res = subprocess.Popen(
        [os.path.dirname(os.path.realpath(__file__)) + '/trec_eval.9.0/./trec_eval', '-m', 'all_trec', qrels_file, qret_file],
        stdout=subprocess.PIPE, shell=False)
  (out, err) = trec_eval_res.communicate()
  trec_eval_res = out.decode("utf-8")
  return trec_eval_res

def getScore(goldX, sysX, metricX, nameX):
  temp_dir        = uuid.uuid4().hex
  qrels_temp_file = '{0}/{1}'.format(temp_dir, 'qrels.txt')
  qret_temp_file  = '{0}/{1}'.format(temp_dir, 'qret.txt')
  res_map         = {}
  try:
    if not os.path.exists(temp_dir):
      os.makedirs(temp_dir)
    else:
      sys.exit("Possible uuid collision")
    format_bioasq2treceval_qrels(goldX, qrels_temp_file)
    format_bioasq2treceval_qret(sysX, nameX, qret_temp_file)
    res_str = trec_evaluate(qrels_temp_file, qret_temp_file)
    for res in res_str.split('\n'):
      res = re.sub(' +', '', res)
      if len(res.split('\t')) == 3:
        res_map[res.split('\t')[0]] = res.split('\t')[2]
  finally:
    os.remove(qrels_temp_file)
    os.remove(qret_temp_file)
    os.rmdir(temp_dir)
  return float(res_map[metricX])

goldf   = sys.argv[1]
sysAf   = sys.argv[2]
sysBf   = sys.argv[3]
metric  = sys.argv[4]

print(goldf)
print(sysAf)
print(sysBf)
print(metric)

with open(goldf) as f:
  gold = json.load(f)

with open(sysAf) as f:
  sysA = json.load(f)

with open(sysBf) as f:
  sysB = json.load(f)

sysA_metric = getScore(gold, sysA, metric, "sysA")
sysB_metric = getScore(gold, sysB, metric, "sysB")
orig_diff   = sysA_metric - sysB_metric

N = 10000
num_invalid = 0

for n in range(N):
  with open(sysAf) as f:
    sysA2 = json.load(f)
  with open(sysBf) as f:
    sysB2 = json.load(f)
  for qi in range(len(sysA2['questions'])):
    rval = random.random()
    if rval < 0.5:
      AD = [d for d in sysA2['questions'][qi]['documents']]
      BD = [d for d in sysB2['questions'][qi]['documents']]
      sysA2['questions'][qi]['documents'] = [d for d in BD]
      sysB2['questions'][qi]['documents'] = [d for d in AD]
  new_sysA_metric = getScore(gold, sysA2, metric, "sysA2")
  new_sysB_metric = getScore(gold, sysB2, metric, "sysB2")
  new_diff = new_sysA_metric - new_sysB_metric
  if new_diff >= orig_diff:
    num_invalid += 1
  if n % 20 == 0 and n > 0:
    print(float(num_invalid +1) / float(n + 1))

print(float(num_invalid + 1) / float(N + 1))

'''
JPDRMM                  
JBERT                   /media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_unfrozen_run_0/all_res_12345.json
JBERT ADAPT             /media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_adapt_unfrozen_run_0/all_res_12345.json
JBERT NF                /media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_run_frozen/all_res_12345.json
JBERT NF ADAPT          /media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_adapt_run_frozen/all_res_12345.json
BERT JPDRMM             /media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_toponly_unfrozen_run_0/all_res_12345.json
BERT JPDRMM ADAPT       /media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_adapt_unfrozen_run_0/all_res_12345.json
BERT JPDRMM NF          /media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_toponly_run_frozen/all_res_12345.json
BERT JPDRMM NF ADAPT    /media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_adapt_run_frozen/all_res_12345.json



/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_unfrozen_run_0_WL_0.1/all_res_12345.json
/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_unfrozen_run_0_WL_0.01/all_res_12345.json

/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_adapt_frozen_run_0_WL_0.0/all_res_12345.json
/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_adapt_frozen_run_0_WL_0.01/all_res_12345.json
/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_adapt_frozen_run_0_WL_0.1/all_res_12345.json
/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_NORESCORE_adapt_frozen_run_0_WL_0.1/all_res_12345.json
/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_NORESCORE_adapt_frozen_run_0_WL_1.0_0.0/all_res_12345.json
/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_snip_adapt_frozen_run_0_WL_0.0/all_res_12345.json


'''


