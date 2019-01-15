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
                ]
            }
        }
    }
    res = es.search(index=index, doc_type=map, body=bod)
    ret = {}
    for item in res['hits']['hits']:
        ret[item[u'_source']['pmid']] = item[u'_score']
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

# q = d['questions'][0]
# pprint(q)
# print(q['body'])
# print(list(t.split('/')[-1] for t in q['documents']))

subm_data = {"questions": []}
for q in tqdm(d['questions']):
    t = {}
    t['body'] = q['body']
    t['id'] = q['id']
    t["snippets"] = []
    t["documents"] = []








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
