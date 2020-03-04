
__author__ = 'Dimitris'

import  numpy as np
from    tqdm import tqdm
import  pickle, re, os, collections, random, json, hnswlib
from    adhoc_vectorizer import get_sentence_vecs

index_dir   = '/media/dpappas/dpappas_data/models_out/batched_semir_bioasq_2020/index/'
bin_fpaths  = [fpath for fpath in os.listdir(index_dir) if fpath.endswith('.bin')]

##############################################################################################################

def retrieve_some_sents():
    ret = []
    for lab, dist in zip(sent_labels1[0].tolist(), sent_distances1[0].tolist()):
        ret.append((dist, lab, labels2names[lab]))
    for lab, dist in zip(sent_labels2[0].tolist(), sent_distances2[0].tolist()):
        ret.append((dist, lab, labels2names[lab]))
    return sorted(ret, key=lambda x: x[0])

##############################################################################################################

dim                 = 50
space               = 'cosine'
max_elements        = 10000000

##############################################################################################################
bioasq_test_set_fpath   = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_1/BioASQ-task8bPhaseA-testset1'
qdata                   = json.load(open(bioasq_test_set_fpath))
questions               = [(q['body'], q['id']) for q in qdata['questions']]
##############################################################################################################

results = {}
pbar = tqdm(bin_fpaths)
for bin_fpath in pbar:
    sent_index_path     = os.path.join(index_dir, bin_fpath)
    labels2names_path   = os.path.join(index_dir, 'labels2names{}.p'.format(bin_fpath.replace('my_index','').replace('.bin','')))
    ####################################################################################
    pbar.set_description("Loading labels2names'{}'\n".format(sent_index_path))
    labels2names        = pickle.load(open(labels2names_path, 'rb'))
    ####################################################################################
    sent_index          = hnswlib.Index(space, dim)
    pbar.set_description("Loading index from '{}'\n".format(sent_index_path))
    sent_index.load_index(sent_index_path, max_elements)
    sent_index.set_num_threads(4)
    ####################################################################################
    for question, qid in questions:
        question_vecs = get_sentence_vecs(question)
        ##################################################################################
        sent_labels1, sent_distances1 = sent_index.knn_query(question_vecs[0], k=25)
        sent_labels2, sent_distances2 = sent_index.knn_query(question_vecs[1], k=25)
        ####################################################################################
        retrieved_sents = retrieve_some_sents()
        if(qid not in results):
            results[qid] = retrieved_sents
        else:
            results[qid].extend(retrieved_sents)
        results[qid] = sorted(results[qid], key=lambda x: x[0])
    ####################################################################################

with open('bioasq_results.json', 'w') as of:
    of.write(json.dumps(results, indent=4, sort_keys=True))
    of.close()

'''

from elasticsearch import Elasticsearch
from pprint import pprint
import json

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

dd  = json.load(open('bioasq_results.json'))

with open('/home/dpappas/elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if(len(line.strip())>0)]
    fp.close()

es = Elasticsearch(cluster_ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
index, doc_type = 'pubmed_abstracts_joint_0_1', 'abstract_map_joint_0_1'
# es.get(index, doc_type, '26749069', params=None)

for qid in dd:
    doc_ids = f7(
        [
            item[2].split(':')[0].strip()
            for item in dd[qid] 
        ]
    )[:10]
    for snip_score, _, snip in dd[qid][:10]:
        pmid, _, snip = snip.split(':',2)
        print(pmid)
        doc_data    = es.get(index, doc_type, pmid)
        title       = doc_data['_source']['joint_text'].split('--------------------')[0].strip()
        abstract    = doc_data['_source']['joint_text'].split('--------------------')[1].strip()
        try:
            ind_from    = title.index(snip)
            ind_to      = ind_from + len(snip) 
            section     = 'title'
        except:
            ind_from    = abstract.index(snip)
            ind_to      = ind_from + len(snip) 
            section     = 'abstract'
    break


'''
