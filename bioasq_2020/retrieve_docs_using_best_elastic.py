
import sys, os, re, json, pickle, ijson
from elasticsearch import Elasticsearch
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from pprint import pprint

# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod = lambda t: re.sub(
    '[.,?;*!%^&_+():-\[\]{}]', '',
    t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()
).split()
bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

doc_index = 'pubmed_abstracts_joint_0_1'

with open('/home/dpappas/elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if(len(line.strip())>0)]
    fp.close()

es = Elasticsearch(cluster_ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

with open('/home/dpappas/bioasq_all/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

print(stopwords)

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

def tokenize(x):
  return bioclean(x)

def GetWords(data, doc_text, words):
  for i in range(len(data['queries'])):
    qwds = tokenize(data['queries'][i]['query_text'])
    for w in qwds:
      words[w] = 1
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      dtext = (
              doc_text[doc_id]['title'] + ' <title> ' + doc_text[doc_id]['abstractText']
              # +
              # ' '.join(
              #     [
              #         ' '.join(mm) for mm in
              #         get_the_mesh(doc_text[doc_id])
              #     ]
              # )
      )
      dwds = tokenize(dtext)
      for w in dwds:
        words[w] = 1

def load_idfs(idf_path, words):
    print('Loading IDF tables')
    #
    # with open(dataloc + 'idf.pkl', 'rb') as f:
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    ret = {}
    for w in words:
        if w in idf:
            ret[w] = idf[w]
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    idf = None
    print('Loaded idf tables with max idf {}'.format(max_idf))
    #
    return ret, max_idf

def get_first_n_1(qtext, n, max_year=2019):
    tokenized_body  = bioclean_mod(qtext)
    tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(tokenized_body)
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool": {
                "must": [{"range": {"DateCompleted": {"gte": "1800", "lte": str(max_year), "format": "dd/MM/yyyy||yyyy"}}}],
                "should": [{"match": {"joint_text": {"query": question, "boost": 1}}}],
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

fpath       = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_1/BioASQ-task8bPhaseA-testset1'
odir        = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_1/bioasq8_bm25_top100/'
test_data   = json.load(open(fpath))

test_docs_to_save = {}
test_data_to_save = {'queries' : []}

for q in tqdm(test_data['questions']):
    qtext       = q['body']
    qid         = q['id']
    #######################################################
    results     = get_first_n_1(qtext, 100)
    #######################################################
    temp_1 = {
        'num_rel'       : 0,
        'num_rel_ret'   : 0,
        'num_ret'       : -1,
        'query_id'      : qid,
        'query_text'    : qtext,
        'relevant_documents': [],
    }
    #######################################################
    all_scores          = [res['_score'] for res in results]
    scaler              = StandardScaler().fit(np.array(all_scores).reshape(-1,1))
    temp_1['num_ret']   = len(all_scores)
    #######################################################
    for res, rank in zip(results, range(1, len(results)+1)):
        test_docs_to_save[res['_id']] = {
            'abstractText'      : res['_source']['joint_text'].split('--------------------', 1)[1].strip(),
            'author'            : '',
            'country'           : '',
            'journalName'       : '',
            'keywords'          : '',
            'meshHeadingsList'  : [],
            'pmid'              : res['_id'],
            'publicationDate'   : res['_source']['DateCompleted'],
            'title'             : res['_source']['joint_text'].split('--------------------')[0].strip()
        }
        #######################################################
        temp_1['relevant_documents'].append({
                'bm25_score'        : res['_score'],
                'doc_id'            : res['_id'],
                'is_relevant'       : False,
                'norm_bm25_score'   : scaler.transform([[res['_score']]])[0][0],
                'rank'              : rank
            })
    test_data_to_save['queries'].append(temp_1)

if(not os.path.exists(odir)):
    os.makedirs(odir)

pickle.dump(test_data_to_save, open(os.path.join(odir, 'bioasq8_bm25_top100.test.pkl'), 'wb'))
pickle.dump(test_docs_to_save, open(os.path.join(odir, 'bioasq8_bm25_docset_top100.test.pkl'), 'wb'))





