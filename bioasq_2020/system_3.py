
__author__ = 'Dimitris'

import  numpy as np
from    tqdm  import tqdm
import  pickle, os, json, re, sys
from adhoc_vectorizer import get_sentence_vecs
# from my_sentence_splitting import get_sents
from nltk.tokenize import sent_tokenize
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def load_idfs(idf_path):
    print('Loading IDF tables')
    #
    # with open(dataloc + 'idf.pkl', 'rb') as f:
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    print('Loaded idf tables with max idf {}'.format(max_idf))
    #
    return idf, max_idf

def tf(term, document):
    tf = 0
    for word in document:
        if word == term:
            tf += 1
    if len(document) == 0:
        return tf
    else:
        return tf/len(document)

def similarity_score(query, document, k1, b, idf_scores, avgdl, normalize, mean, deviation, rare_word):
    score = 0
    for query_term in query:
        if query_term not in idf_scores:
            score += rare_word * (
                    (tf(query_term, document) * (k1 + 1)) /
                    (
                            tf(query_term, document) +
                            k1 * (1 - b + b * (len(document) / avgdl))
                    )
            )
        else:
            score += idf_scores[query_term] * ((tf(query_term, document) * (k1 + 1)) / (tf(query_term, document) + k1 * (1 - b + b * (len(document) / avgdl))))
    if normalize:
        return ((score - mean)/deviation)
    else:
        return score

def tokenize(x):
  return bioclean(x)

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

in_dir      = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_1/bioasq8_bm25_top100/'

docs_data   = pickle.load(open(os.path.join(in_dir, 'bioasq8_bm25_docset_top100.test.pkl'), 'rb'))
ret_data    = json.load(open("/home/dpappas/bioasq8_batch1_out_jpdrmm/v3 test_emit_bioasq.json"))

avgdl, mean, deviation      = 21.1907, 0.6275, 1.2210
print(avgdl, mean, deviation)
###########################################################
print('loading idfs')
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
idf, max_idf        = load_idfs(idf_pickle_path)
###########################################################

system_subm_data    = {'questions':[]}

for quer in tqdm(ret_data['queries']):
    qid         = quer['id']
    qtext       = quer['body']
    quest_toks  = tokenize(qtext)
    qvecs       = get_sentence_vecs(qtext)
    if(qvecs is None):
        continue
    #############################################
    subm_data = {
        "body"      : qtext,
        "documents" : [],
        "id"        : qid,
        "snippets"  : []
    }
    #############################################
    sent_res    = []
    #############################################
    for ret_doc in quer['documents']:
        norm_bm25   = 1
        doc_id      = ret_doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '').strip()
        #############################################
        abstract    = ' '.join([ token for token in docs_data[doc_id]['abstractText'].split() if(not token.startswith('__') and not token.endswith('__'))])
        abs_sents   = sent_tokenize(abstract)
        title       = ' '.join([token for token in docs_data[doc_id]['title'].split() if(not token.startswith('__') and not token.endswith('__'))])
        tit_sents   = sent_tokenize(title)
        #############################################
        for sent in tit_sents:
            svecs       = get_sentence_vecs(sent)
            if (svecs is None):
                continue
            sim         = cosine_similarity(qvecs, svecs).max()
            offset_from = title.index(sent)
            offset_to   = offset_from + len(sent)
            sent_bm25   = similarity_score(quest_toks, tokenize(sent), 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
            sent_res.append(
                (sim, doc_id, norm_bm25, 'title', sent, sent_bm25, offset_from, offset_to)
            )
        for sent in abs_sents:
            svecs       = get_sentence_vecs(sent)
            if (svecs is None):
                continue
            sim         = cosine_similarity(qvecs, svecs).max()
            offset_from = abstract.index(sent)
            offset_to   = offset_from + len(sent)
            sent_bm25   = similarity_score(quest_toks, tokenize(sent), 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
            sent_res.append(
                (sim, doc_id, norm_bm25, 'abstract', sent, sent_bm25, offset_from, offset_to)
            )
    #############################################
    sent_res    = sorted(sent_res, key=lambda x: x[0] * x[2] * x[5], reverse=True)
    doc_res     = f7([t[1] for t in sent_res])
    #############################################
    subm_data['documents']  = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(doc_id) for doc_id in doc_res[:10]]
    subm_data[ "snippets"]  = [
        {
            "beginSection"          : s_res[3],
            "document"              : "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(s_res[1]),
            "endSection"            : s_res[3],
            "offsetInBeginSection"  : s_res[6],
            "offsetInEndSection"    : s_res[7],
            "text"                  : s_res[4]
        }
        for s_res in sent_res[:10]
    ]
    system_subm_data['questions'].append(subm_data)

###########################################################

odir = '/home/dpappas/bioasq8_batch1_system3_out/'
if (not os.path.exists(odir)):
    os.makedirs(odir)

with open(os.path.join(odir, 'bioasq8_batch1_system3_results.json'), 'w') as f:
    f.write(json.dumps(system_subm_data, indent=4, sort_keys=True))

