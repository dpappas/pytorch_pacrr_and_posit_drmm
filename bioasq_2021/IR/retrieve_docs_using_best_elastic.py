
import sys, os, re, json, pickle, ijson
from elasticsearch import Elasticsearch
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from pprint import pprint
import sys

def fix2(qtext):
    qtext = qtext.lower()
    if(qtext.startswith('can ')):
        qtext = qtext[4:]
    if(qtext.startswith('list the ')):
        qtext = qtext[9:]
    if(qtext.startswith('list ')):
        qtext = qtext[5:]
    if(qtext.startswith('describe the ')):
        qtext = qtext[13:]
    if(qtext.startswith('describe ')):
        qtext = qtext[9:]
    if('list as many ' in qtext and 'as possible' in qtext):
        qtext = qtext.replace('list as many ', '')
        qtext = qtext.replace('as possible', '')
    if('yes or no' in qtext):
        qtext = qtext.replace('yes or no', '')
    if('also known as' in qtext):
        qtext = qtext.replace('also known as', '')
    if('is used to ' in qtext):
        qtext = qtext.replace('is used to ', '')
    if('are used to ' in qtext):
        qtext = qtext.replace('are used to ', '')
    tokenized_body  = [t for t in qtext.split() if t not in stopwords]
    tokenized_body  = bioclean_mod(' '.join(tokenized_body))
    question        = ' '.join(tokenized_body)
    return question

def fix1(qtext):
    tokenized_body  = bioclean_mod(qtext)
    tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(tokenized_body)
    return question

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

stopwords.add('with')
print(stopwords)

def tokenize(x):
  return bioclean(x)

def get_first_n_1(qtext, n, max_year=2022):
    # tokenized_body  = bioclean_mod(qtext)
    # tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    # question        = ' '.join(tokenized_body)
    question = fix2(qtext)
    print(question)
    ################################################
    bod             = {
        "size": n,
        "query": {
            "bool": {
                "must": [{"range": {"DateCompleted": {"gte": "1900", "lte": str(max_year), "format": "dd/MM/yyyy||yyyy"}}}],
                "should": [
                    {
                        "match": {
                            "joint_text": {
                                "query": question,
                                "boost": 1,
                                'minimum_should_match': "30%"
                            }
                        }
                    },
                    {
                        "match": {
                            "joint_text": {
                                "query": question,
                                "boost": 1,
                                'minimum_should_match': "50%"
                            }
                        }
                    },
                    {
                        "match": {
                            "joint_text": {
                                "query": question,
                                "boost": 1,
                                'minimum_should_match': "70%"
                            }
                        }
                    },
                    {"match_phrase": {"joint_text": {"query": question, "boost": 1}}}
                ],
                "minimum_should_match": 1,
            }
        }
    }
    res             = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

batch       = int(sys.argv[1])
# fpath       = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_{}/BioASQ-task8bPhaseA-testset{}'.format(batch,batch)
# odir        = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_{}/bioasq8_bm25_top100/'.format(batch)
# fpath       = '/home/dpappas/bioasq_2021/BioASQ-task9bPhaseA-testset{}'.format(batch,batch)
# odir        = '/home/dpappas/bioasq_2021/test_batch_{}/bm25_top100/'.format(batch)
fpath       = '/home/dpappas/bioasq_2021/BioASQ-task9bPhaseA-testset{}'.format(batch,batch)
odir        = '/home/dpappas/bioasq_2021/test_batch_{}/bm25_top100/'.format(batch)
test_data   = json.load(open(fpath))

test_docs_to_save = {}
test_data_to_save = {'queries' : []}

for q in tqdm(test_data['questions']):
    qtext       = q['body']
    print(qtext)
    qid         = q['id']
    #######################################################
    results     = get_first_n_1(qtext, 100)
    print([t['_id'] for t in results])
    #######################################################
    temp_1      = {
        'num_rel'               : 0,
        'num_rel_ret'           : 0,
        'num_ret'               : -1,
        'query_id'              : qid,
        'query_text'            : qtext,
        'relevant_documents'    : [],
        'retrieved_documents'   : []
    }
    #######################################################
    all_scores          = [res['_score'] for res in results]
    # print(all_scores)
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
        temp_1['retrieved_documents'].append({
                'bm25_score'        : res['_score'],
                'doc_id'            : res['_id'],
                'is_relevant'       : False,
                'norm_bm25_score'   : scaler.transform([[res['_score']]])[0][0],
                'rank'              : rank
            })
    test_data_to_save['queries'].append(temp_1)
    # break

if(not os.path.exists(odir)):
    os.makedirs(odir)

pickle.dump(test_data_to_save, open(os.path.join(odir, 'bioasq9_bm25_top100.test.pkl'), 'wb'))
pickle.dump(test_docs_to_save, open(os.path.join(odir, 'bioasq9_bm25_docset_top100.test.pkl'), 'wb'))


'''
source /home/dpappas/venvs/elasticsearch_old/bin/activate
python retrieve_docs.py 4
'''


