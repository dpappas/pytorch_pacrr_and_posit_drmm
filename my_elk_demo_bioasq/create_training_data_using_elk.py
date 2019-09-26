
import sys, os, re, json, pickle, ijson
from elasticsearch import Elasticsearch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pprint import pprint

# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()
bioclean        = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

doc_index       = 'pubmed_abstracts_joint_0_1'

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

def load_all_data(dataloc, idf_pickle_path):
    print('loading pickle data')
    #
    with open(dataloc +'trainining7b.json', 'r') as f:
        bioasq7_data = json.load(f)
        bioasq7_data = dict((q['id'], q) for q in bioasq7_data['questions'])
    #
    with open(dataloc + 'bioasq7_bm25_top100.dev.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_docset_top100.dev.pkl', 'rb') as f:
        dev_docs = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_top100.train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_docset_top100.train.pkl', 'rb') as f:
        train_docs = pickle.load(f)
    print('loading words')
    #
    words               = {}
    GetWords(train_data, train_docs, words)
    GetWords(dev_data,   dev_docs,   words)
    #
    print('loading idfs')
    idf, max_idf    = load_idfs(idf_pickle_path, words)
    return dev_data, dev_docs, train_data, train_docs, idf, max_idf, bioasq7_data

def get_first_n_1(qtext, n, max_year=2017):
    tokenized_body = bioclean_mod(qtext)
    tokenized_body = [t for t in tokenized_body if t not in stopwords]
    question = ' '.join(tokenized_body)
    ################################################
    bod = {
        "size": n,
        "query": {
            "bool": {
                "must": [
                    {"range": {"DateCompleted": {"gte": "1800", "lte": str(max_year), "format": "dd/MM/yyyy||yyyy"}}}],
                "should": [{"match": {"joint_text": {"query": question, "boost": 1}}}],
                "minimum_should_match": 1,
            }
        }
    }
    res = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

def put_b_k1(b, k1):
    print(es.indices.close(index=doc_index))
    print(es.indices.put_settings(
        index=doc_index,
        body={
            "similarity": {
                "my_similarity": {
                    "type": "BM25",
                    "b": b,
                    "k1": k1
                }
            }
        }
    ))
    print(es.indices.open(index=doc_index))

def get_new(data):
    new_data = []
    new_docs = {}
    for q in tqdm(data['queries'][1840:]):
        hits        = get_first_n_1(q['query_text'], 100, max_year=2018)
        ret_pmids   = set(hit['_source']['pmid'] for hit in hits)
        num_ret     = len(hits)
        num_rel_ret = len(ret_pmids.intersection(q['relevant_documents']))
        datum = {
            'query_text'            : q['query_text'],
            'ret_pmids'             : ret_pmids,
            'num_ret'               : num_ret,
            'num_rel'               : q['num_rel'],
            'num_rel_ret'           : num_rel_ret,
            'query_id'              : q['query_id'],
            'relevant_documents'    : q['relevant_documents'],
            'retrieved_documents'   : []
        }
        all_mb25s   = [[hit['_score']] for hit in hits]
        if(len(all_mb25s)):
            all_mb25s = all_mb25s + all_mb25s
        scaler      = StandardScaler()
        scaler2     = MinMaxScaler()
        scaler.fit(all_mb25s)
        scaler2.fit(all_mb25s)
        print(scaler.mean_)
        for hit, rank in zip(hits, range(1, len(hits)+1)):
            datum['retrieved_documents'].append(
                {
                  'bm25_score'                  : hit['_score'],
                  'doc_id'                      : hit['_source']['pmid'],
                  'is_relevant'                 : hit['_source']['pmid'] in q['relevant_documents'],
                  'norm_bm25_score_standard'    : scaler.transform([[hit['_score']]])[0][0],
                  'norm_bm25_score_minmax'      : scaler2.transform([[hit['_score']]])[0][0],
                  'rank'                        : rank
                }
            )
            new_docs[hit['_source']['pmid']] = {
                'title'             : hit['_source']['joint_text'].split('--------------------')[0].strip(),
                'abstractText'      : hit['_source']['joint_text'].split('--------------------')[1].strip(),
                'keywords'          : hit['_source']['Keywords'],
                'meshHeadingsList'  : hit['_source']['MeshHeadings'],
                'chemicals'         : hit['_source']['Chemicals'],
                'pmid'              : hit['_source']['pmid'],
                'publicationDate'   : hit['_source']['DateCompleted']
            }
        new_data.append(datum)
    return new_data, new_docs

with open('/home/dpappas/elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if (len(line.strip()) > 0)]
    fp.close()

es = Elasticsearch(cluster_ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

dataloc = '/home/dpappas/bioasq_all/bioasq7_data/'
idf_pickle_path = '/home/dpappas/bioasq_all/idf.pkl'
(dev_data, dev_docs, train_data, train_docs, idf, max_idf, bioasq7_data) = load_all_data(dataloc, idf_pickle_path)

with open('/home/dpappas/bioasq_all/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

print(stopwords)

# b, k1   = 0.3, 0.6
# put_b_k1(b, k1)

odir_dataloc = '/home/dpappas/bioasq_all/bioasq7_data_demo/data/'

new_train_data, new_train_docs  = get_new(train_data)
new_dev_data,   new_dev_docs    = get_new(dev_data)

with open(odir_dataloc + 'trainining7b.json', 'w') as f:
    f.write(json.dumps(bioasq7_data, indent=4, sort_keys=True))

pickle.dump(new_dev_data,   open(odir_dataloc + 'bioasq7_bm25_top100.dev.pkl', 'wb'))
pickle.dump(new_dev_docs,   open(odir_dataloc + 'bioasq7_bm25_docset_top100.dev.pkl', 'wb'))
pickle.dump(new_train_data, open(odir_dataloc + 'bioasq7_bm25_top100.train.pkl', 'wb'))
pickle.dump(new_train_docs, open(odir_dataloc + 'bioasq7_bm25_docset_top100.train.pkl', 'wb'))



