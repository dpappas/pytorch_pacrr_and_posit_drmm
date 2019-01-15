import zipfile
import json
from pprint import pprint
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

def get_elk_results(search_text):
    bod = {
        "_source": ["ArticleTitle", "pmid"],
        "size": 100,
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
    res = es.search(index=index, doc_type=map, body=bod)
    ret = {}
    for item in res['hits']['hits']:
        ret[
            'http://www.ncbi.nlm.nih.gov/pubmed/{}'.format(item[u'_source']['pmid'])
        ] = item[u'_score']
    return ret

archive = zipfile.ZipFile('/home/dpappas/Downloads/BioASQ-training7b.zip', 'r')
jsondata = archive.read('BioASQ-training7b/trainining7b.json')
d = json.loads(jsondata)

maxx = 0
for q in tqdm(d['questions']):
    for link in q['documents']:
        t = int(link.split('/')[-1])
        maxx = max([maxx, t])

print('https://www.ncbi.nlm.nih.gov/pubmed/{}'.format(maxx))

es = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
index = 'pubmed_abstracts_0_1'
map = "abstract_map_0_1"

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

emited_fpath = '/home/dpappas/elk_doc_ret_emit.json'
with open(emited_fpath, 'w') as f:
    f.write(json.dumps(subm_data, indent=4, sort_keys=True))
    f.close()


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
