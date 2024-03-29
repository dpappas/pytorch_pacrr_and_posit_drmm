

import  sys, os, re, json, pickle, ijson
from    tqdm import tqdm
from    sklearn.preprocessing import StandardScaler
import  numpy as np
from    elasticsearch import Elasticsearch
from    pprint import pprint
import  torch, nltk
import  datetime
from    textblob import TextBlob
from    nltk import ngrams

#####################################################################################
# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()
bioclean        = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
#####################################################################################

with open('/home/dpappas/bioasq_all/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

stopwords   = stopwords.union(set(nltk.corpus.stopwords.words("english")))
stopwords.add('what')
stopwords.add('who')
stopwords.add('which')
stopwords.add('know')
print(stopwords)
#####################################################################################

with open('/home/dpappas/elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if(len(line.strip())>0)]
    fp.close()

es = Elasticsearch(cluster_ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

#####################################################################################

my_seed     = 1989
use_cuda    = torch.cuda.is_available()
if(use_cuda):
    torch.cuda.manual_seed(my_seed)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu   = torch.cuda.device_count()

#####################################################################################

def tokenize(x):
  return bioclean(x)

def keep_only_longest(phrases):
    ret = []
    for phrase in sorted(phrases, key=lambda x : len(x), reverse=True):
        if(any(phrase in t for t in ret)):
            continue
        else:
            ret.append(phrase)
    return ret

def get_noun_chunks(text):
    blob    = TextBlob(text)
    pt      = blob.pos_tags
    nps     = []
    for i in range(1,5):
        nps.extend(
            [
                ' '.join([tt[0] for tt in gr])
                for gr in ngrams(pt, i)
                if(
                    all(
                        (t[1] in ['NNP','NN','JJ','ADJ','ADV'] and t[0].lower() not in stopwords)
                        for t in gr
                    )
                )
            ]
        )
    ret = list(blob.noun_phrases)+nps
    ret = keep_only_longest(ret)
    return ret

def retrieve_some_docs(qtext, n=100, exclude_pmids=None):
    noun_chunks     = get_noun_chunks(qtext)
    tokenized_body  = bioclean_mod(qtext)
    tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(tokenized_body)
    '''
    bod             = {
        'size' : n,
        "query": {
            "bool" : {
                "should" : [{"match": {"joint_text": {"query": question}}}] + [
                    {"match_phrase": {"joint_text": {"query": chunk}}}
                    for chunk in noun_chunks
                ],
                "minimum_should_match" : 1,
                "boost" : 1.0
            }
        }
    }
    '''
    ################################################
    bod             = {
        "size": n,
        "query": {
            "bool": {
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
    ################################################
    if(exclude_pmids):
        # bod["query"]["bool"]["must_not"] = [{"_id": {"values": exclude_pmids}}]
        bod["query"]["bool"]["must_not"] = [
            {
                "ids": {
                    # "type" : "_doc",
                    "values": exclude_pmids
                }
            },
            {"terms" : { "doc_id" : exclude_pmids }},
            {"terms" : { "doc.cord_uid" : exclude_pmids }}
        ]
    res = es.search(index=index, body=bod)
    return res

#####################################################################################

# index   = 'allenai_covid_index_2020_11_29_csv'
index   = 'allenai_covid_index_2021_01_25_csv'
es      = Elasticsearch(['127.0.0.1:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

#####################################################################################

def get_first_n(qtext, n, exclude_pmids=None):
    results = retrieve_some_docs(qtext, n=n, exclude_pmids=exclude_pmids)['hits']['hits']
    #######################################################
    temp_1 = {
        'num_rel': 0,
        'num_rel_ret': 0,
        'num_ret': -1,
        'query_id': 1234567890,
        'query_text': qtext,
        'relevant_documents': [],
        'retrieved_documents': []
    }
    #######################################################
    all_scores = [res['_score'] for res in results]
    # print(all_scores)
    if(len(all_scores)==0):
        return temp_1
    scaler = StandardScaler().fit(np.array(all_scores).reshape(-1, 1))
    temp_1['num_ret'] = len(all_scores)
    #######################################################
    for res, rank in zip(results, range(1, len(results) + 1)):
        temp_1['retrieved_documents'].append({
            'bm25_score'        : res['_score'],
            'doc_id'            : res['_id'],
            'is_relevant'       : False,
            'norm_bm25_score'   : scaler.transform([[res['_score']]])[0][0],
            'rank'              : rank,
            'doc'               : {
                'title'             : res['_source']['joint_text'].split('--------------------', 1)[0].strip(),
                'abstractText'      : res['_source']['joint_text'].split('--------------------', 1)[1].strip(),
                'pmid'              : res['_id'],
                'cord_uid'          : res['_source']['cord_uid'],
                'publish_time'      : res['_source']['publish_time']
            }
        })
    return temp_1

if __name__ == '__main__':
    qtext   = 'Which diagnostic test is approved for coronavirus infection screening?'
    res     = get_first_n(qtext, n=100, exclude_pmids=['6crputzl'])
    t1      = [t['doc_id'] for t in res['retrieved_documents']]
    pprint(t1)
    res     = get_first_n(qtext, n=100, exclude_pmids=['aspx7cc6'])
    t2      = [t['doc_id'] for t in res['retrieved_documents']]
    pprint(t2)
    pprint(set(t1)-set(t2))
    pprint(set(t2)-set(t1))
    pprint(get_first_n("Which age group and gender are more susceptible of developing a Kawasaki-like syndrome with Covid-19?", n=100, exclude_pmids=["tnouw1h0", "ukmbo7mn", "5g5kreqz", "rdgknsps", "0xgjpd80", "vfbp2psw", "quc4s5wp", "5dyhyx9a", "6vy2tasf", "budtkxh5"]))




