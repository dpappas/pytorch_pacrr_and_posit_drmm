
__author__ = 'Dimitris'

import  json
import  numpy                       as np
from    tqdm                        import tqdm
import  pickle, os
import  hnswlib
import  adhoc_vectorizer

##############################################################################################################

def print_some_sents(max_dist_sent):
    global printed_one
    print(20 * '-')
    for lab, dist in zip(sent_labels1[0].tolist(), sent_distances1[0].tolist()):
        if(dist<=max_dist_sent):
            print('{} ~ {} ~ {}'.format(dist, lab, labels2names[lab]))
    print(20 * '-')
    for lab, dist in zip(sent_labels2[0].tolist(), sent_distances2[0].tolist()):
        if(dist<=max_dist_sent):
            print('{} ~ {} ~ {}'.format(dist, lab, labels2names[lab]))

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
bioasq_test_set_fpath   = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseA-testset1'
qdata                   = json.load(open(bioasq_test_set_fpath))
questions               = [q['body'] for q in  qdata['questions']]
##############################################################################################################

# question = '''In CT imaging for the diagnosis or staging of acute diverticulitis what is the test accuracy of CT imaging for the diagnosis or staging of acute diverticulitis ?'''
# question = '''what is the test accuracy of CT imaging for the diagnosis or staging of acute diverticulitis ?'''

##############################################################################################################

max_dist = 0.05
results = {}
for bin_fpath in tqdm(bin_fpaths):
    # bin_fpath           = bin_fpaths[0]
    sent_index_path     = os.path.join(index_dir, bin_fpath)
    labels2names_path   = os.path.join(index_dir, 'labels2names{}.p'.format(bin_fpath.replace('my_index','').replace('.bin','')))
    ####################################################################################
    print("\nLoading labels2names'{}'\n".format(sent_index_path))
    labels2names        = pickle.load(open(labels2names_path, 'rb'))
    ####################################################################################
    sent_index          = hnswlib.Index(space, dim)
    print("\nLoading index from '{}'\n".format(sent_index_path))
    sent_index.load_index(sent_index_path, max_elements)
    sent_index.set_num_threads(4)
    ####################################################################################
    for question in questions:
        question_vecs   = np.stack([get_vec(tok) for tok in bioclean(question)], 0)
        sent_mask       = np.ones(question_vecs.shape[0])
        question_vecs   = model.encode_sent([question_vecs], [sent_mask]).squeeze(0).cpu().data.numpy()
        ####################################################################################
        sent_labels1, sent_distances1 = sent_index.knn_query(question_vecs[0], k=25)
        sent_labels2, sent_distances2 = sent_index.knn_query(question_vecs[1], k=25)
        ####################################################################################
        # print_some_sents(max_dist_sent=max_dist)
        retrieved_sents = retrieve_some_sents()
        if(question not in results):
            results[question] = retrieved_sents
        else:
            results[question].extend(retrieved_sents)
        results[question] = sorted(results[question], key=lambda x: x[0])
    ####################################################################################

with open('bioasq_results.json', 'w') as of:
    of.write(json.dumps(results, indent=4, sort_keys=True))
    of.close()

import json

dd  = json.load(open('bioasq_results.json'))

dd3 = {}
for question, v in dd.items():
    if(question not in dd3):
        dd3[question] = {}
    for score, _, ss in v:
        pmid = ss.split(':')[0]
        try:
            dd3[question][pmid] += (1-score)
        except:
            dd3[question][pmid] = (1-score)
    dd3[question] = sorted(list(dd3[question].items()), key=lambda x:x[1], reverse=True)[:10]

with open('bioasq_results_scores.json', 'w') as of:
    of.write(json.dumps(dd3, indent=4, sort_keys=True))
    of.close()






