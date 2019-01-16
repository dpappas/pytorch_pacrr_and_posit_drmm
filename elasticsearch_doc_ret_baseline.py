import os
import zipfile
import json
from pprint import pprint
from tqdm import tqdm
import subprocess
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

def print_the_results(fpath_gold, fpath_emit):
    bioasq_snip_res = get_bioasq_res(fpath_gold, fpath_emit)
    print('MAP documents: {}'.format(bioasq_snip_res['MAP documents']))
    print('F1 snippets: {}'.format(bioasq_snip_res['F1 snippets']))
    print('MAP snippets: {}'.format(bioasq_snip_res['MAP snippets']))
    print('GMAP snippets: {}'.format(bioasq_snip_res['GMAP snippets']))
    #

def get_bioasq_res(fpath_gold, fpath_emit):
    '''
    java -Xmx10G -cp /home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar
    evaluation.EvaluatorTask1b -phaseA -e 5
    /home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/BioASQ-task6bPhaseB-testset1
    ./drmm-experimental_submit.json
    '''
    jar_path = retrieval_jar_path
    #
    command = ['java', '-Xmx10G', '-cp', jar_path, 'evaluation.EvaluatorTask1b', '-phaseA', '-e', '5', fpath_gold,
               fpath_emit]
    print(' '.join(command))
    bioasq_eval_res = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
    (out, err) = bioasq_eval_res.communicate()
    lines = out.decode("utf-8").split('\n')
    ret = {}
    for line in lines:
        if (':' in line):
            k = line.split(':')[0].strip()
            v = line.split(':')[1].strip()
            ret[k] = float(v)
    return ret

def create_body(search_text):
    return {
        "size": 100,
        "_source": [
            "ArticleTitle",
            "pmid"
        ],
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": search_text,
                            "type": "best_fields",
                            "fields": [
                                "ArticleTitle",
                                "AbstractText"
                            ],
                            "minimum_should_match": "50%",
                            "slop": 2
                        }
                    }
                ],
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "lte": "01/04/2018",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ]
            }
        }
    }

def create_body_1(search_text):
    return {
        "size": 100,
        "_source": ["pmid"],
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": search_text,
                            "type": "best_fields",
                            "fields": ["ArticleTitle", "AbstractText"],
                            "minimum_should_match": "50%",
                            "slop": 2
                        }
                    }
                ],
                "filter": [
                    {"range": {"DateCompleted": {"lte": "01/04/2018", "format": "dd/MM/yyyy||yyyy"}}},
                    {"exists": {"field": "ArticleTitle"}},
                    {"exists": {"field": "AbstractText"}},
                    {"regexp": {"ArticleTitle": ".+"}},
                    {"regexp": {"AbstractText": ".+"}}
                ]
            }
        }
    }

def create_body_2(search_text):
    return {
        "size": 100,
        "_source": ["pmid"],
        "query": {
            "bool": {
                "should": [
                    {"match_phrase": {"message": {"query": search_text}}},
                ],
                "filter": [
                    {"range": {"DateCompleted": {"lte": "01/04/2018", "format": "dd/MM/yyyy||yyyy"}}},
                    {"exists": {"field": "ArticleTitle"}},
                    {"exists": {"field": "AbstractText"}},
                    {"regexp": {"ArticleTitle": ".+"}},
                    {"regexp": {"AbstractText": ".+"}}
                ]
            }
        }
    }

def create_body_3(search_text):
    return {
        "size": 100,
        "_source": ["pmid"],
        "query": {
            "bool": {
                "should": [
                    {
                        "more_like_this": {
                            "fields": ["title", "description"],
                            "like": search_text,
                            "min_term_freq": 1,
                            "max_query_terms": 25
                        }
                    }
                ],
                "filter": [
                    {"range": {"DateCompleted": {"lte": "01/04/2018", "format": "dd/MM/yyyy||yyyy"}}},
                    {"exists": {"field": "ArticleTitle"}},
                    {"exists": {"field": "AbstractText"}},
                    {"regexp": {"ArticleTitle": ".+"}},
                    {"regexp": {"AbstractText": ".+"}}
                ]
            }
        }
    }

def create_body_4(search_text):
    return {
        "size": 100,
        "_source": ["pmid"],
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": search_text,
                            "type": "best_fields",
                            "fields": ["ArticleTitle", "AbstractText"],
                            "minimum_should_match": "50%",
                            "slop": 2
                        }
                    }
                ]
            }
        },
        "filter": [
            {"range": {"DateCompleted": {"lte": "01/04/2018", "format": "dd/MM/yyyy||yyyy"}}},
            {"exists": {"field": "ArticleTitle"}},
            {"exists": {"field": "AbstractText"}},
            {"regexp": {"ArticleTitle": ".+"}},
            {"regexp": {"AbstractText": ".+"}}
        ]
    }


def create_body_5(search_text):
    return {
        "size": 100,
        "_source": ["pmid"],
        "query": {
            "bool": {
                "should": [
                    {"match_phrase": {"message": {"query": search_text}}},
                ],
                "filter": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "01/01/1985",
                                "lte": "01/04/2018",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    },
                    {"exists": {"field": "ArticleTitle"}},
                    {"exists": {"field": "AbstractText"}},
                    {"regexp": {"ArticleTitle": ".+"}},
                    {"regexp": {"AbstractText": ".+"}}
                ]
            }
        }
    }

def get_elk_results(search_text):
    bod = create_body(search_text)
    res = es.search(index=index, doc_type=map, body=bod)
    ret = {}
    for item in res['hits']['hits']:
        ret[
            'http://www.ncbi.nlm.nih.gov/pubmed/{}'.format(item[u'_source']['pmid'])
        ] = item[u'_score']
    return ret

retrieval_jar_path = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'
gold_fpath = '/home/dpappas/bioasq_all/BioASQ-training7b/trainining7b.json'
elk_ip = 'harvester2.ilsp.gr:9200'

emited_fpath_elastic = '/home/dpappas/elk_doc_ret_emit.json'
emited_fpath_galago = '/home/dpappas/elk_doc_ret_emit_galago.json'
gold_annot_fpath = '/home/dpappas/elk_doc_ret_gold.json'
galago_ret_file = '/home/dpappas/bioasq7_bm25_retrieval.train.txt'

archive = zipfile.ZipFile('/home/dpappas/bioasq_all/BioASQ-training7b.zip', 'r')
jsondata = archive.read('BioASQ-training7b/trainining7b.json')
d = json.loads(jsondata)
#
maxx = 0
minn = 10000000
for q in tqdm(d['questions']):
    for link in q['documents']:
        t = int(link.split('/')[-1])
        maxx = max([maxx, t])
        minn = min([minn, t])

print(minn)
print(maxx)

if (not os.path.exists(emited_fpath_elastic)):
    archive = zipfile.ZipFile('/home/dpappas/bioasq_all/BioASQ-training7b.zip', 'r')
    jsondata = archive.read('BioASQ-training7b/trainining7b.json')
    d = json.loads(jsondata)
    #
    maxx = 0
    for q in tqdm(d['questions']):
        for link in q['documents']:
            t = int(link.split('/')[-1])
            maxx = max([maxx, t])
    #
    es = Elasticsearch([elk_ip], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    index = 'pubmed_abstracts_0_1'
    map = "abstract_map_0_1"
    #
    subm_data = {"questions": []}
    for q in tqdm(d['questions']):
        t = {
            'body': q['body'],
            'id': q['id'],
            'snippets': [],
            'documents': []
        }
        #
        elk_scored_pmids = get_elk_results(q['body'])
        sorted_keys = sorted(elk_scored_pmids.keys(), key=lambda x: elk_scored_pmids[x], reverse=True)
        t['documents'] = sorted_keys
        subm_data['questions'].append(t)
    with open(emited_fpath_elastic, 'w') as f:
        f.write(json.dumps(subm_data, indent=4, sort_keys=True))
        f.close()

if (not os.path.exists(gold_annot_fpath)):
    archive = zipfile.ZipFile('/home/dpappas/bioasq_all/BioASQ-training7b.zip', 'r')
    jsondata = archive.read('BioASQ-training7b/trainining7b.json')
    d = json.loads(jsondata)
    gdata = {"questions": []}
    for q in tqdm(d['questions']):
        t = {
            'body': q['body'],
            'id': q['id'],
            'snippets': [],
            'documents': q['documents']
        }
        gdata['questions'].append(t)
    with open(gold_annot_fpath, 'w') as f:
        f.write(json.dumps(gdata, indent=4, sort_keys=True))
        f.close()

if (not os.path.exists(emited_fpath_galago)):
    with open(galago_ret_file) as f:
        lines = [l.strip() for l in f.readlines() if (len(l.strip()) > 0)]
        f.close()
    ####
    ret = {}
    for l in lines:
        spl = l.split()
        qid = spl[0]
        pmid = spl[2]
        score = float(spl[-2])
        if (qid in ret):
            ret[qid][pmid] = score
        else:
            ret[qid] = {}
            ret[qid][pmid] = score
    ####
    subm_data = {"questions": []}
    for qid in ret.keys():
        t = {
            'body': '',
            'id': qid,
            'snippets': [],
            'documents': [
                'http://www.ncbi.nlm.nih.gov/pubmed/{}'.format(t) for t in
                sorted(ret[qid].keys(), key=lambda x: ret[qid][x], reverse=True)
            ]
        }
        subm_data['questions'].append(t)
    ####
    with open(emited_fpath_galago, 'w') as f:
        f.write(json.dumps(subm_data, indent=4, sort_keys=True))
        f.close()

print_the_results(gold_annot_fpath, emited_fpath_elastic)
print_the_results(gold_annot_fpath, emited_fpath_galago)

print_the_results(gold_annot_fpath, '/home/dpappas/elk_doc_ret_emit.json')
print_the_results(gold_annot_fpath, '/home/dpappas/elk_doc_ret_emit_1.json')
print_the_results(gold_annot_fpath, '/home/dpappas/elk_doc_ret_emit_2.json')
print_the_results(gold_annot_fpath, '/home/dpappas/elk_doc_ret_emit_3.json')
print_the_results(gold_annot_fpath, emited_fpath_galago)

'''

GET pubmed_abstracts_0_1/_search
{
  "_source": ["ArticleTitle", "pmid"],
  "size" : 100,
  "query": {
    "bool": {
      "should": [
        {
          "multi_match" : {
            "query"                 : "What is Mendelian randomization",
            "type"                  : "best_fields",
            "fields"                : ["ArticleTitle", "AbstractText"],
            "minimum_should_match"  : "50%",
            "slop"                  : 2
          }
        }
      ]
    }
  }
}

'''

'''
python3.6 \
/home/DATA/Biomedical/document_ranking/bioasq_data/document_retrieval/queries2galago.py \
/home/dpappas/bioasq_all/BioASQ-training7b/trainining7b.json /home/dpappas/trolololo.json

/home/DATA/Biomedical/document_ranking/bioasq_data/document_retrieval/galago-3.10-bin/bin/galago \
batch-search \
--index=pubmed_only_abstract_galago_index \
--verbose=False \
--requested=100 \
--scorer=bm25 \
--defaultTextPart=postings.krovetz \
--mode=threaded \
/home/dpappas/trolololo.json \
> \
/home/dpappas/bioasq7_bm25_retrieval.train.txt

python3.6 \
/home/DATA/Biomedical/document_ranking/bioasq_data/document_retrieval/generate_bioasq_data.py \
/home/dpappas/bioasq7_bm25_retrieval.train.txt

'''
