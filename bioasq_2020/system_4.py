
__author__ = 'Dimitris'

import  numpy as np
from    tqdm import tqdm
import  pickle, re, os, collections, random, json, hnswlib
from    adhoc_vectorizer import get_sentence_vecs

b                       = '2'
index_dir               = '/media/dpappas/dpappas_data/models_out/batched_semir_bioasq_2020/index/'
bioasq_test_set_fpath   = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_{}/BioASQ-task8bPhaseA-testset{}'.format(b, b)
odir                    = '/home/dpappas/bioasq_2020/system4_output_b{}/'.format(b)

if(not os.path.exists(odir)):
    os.makedirs(odir)

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

with open(os.path.join(odir, 'bioasq_results.json'), 'w') as of:
    of.write(json.dumps(results, indent=4, sort_keys=True))
    of.close()
