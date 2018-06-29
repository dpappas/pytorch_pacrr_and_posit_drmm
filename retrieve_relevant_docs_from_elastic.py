
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from pprint import pprint

index   = 'pubmed_abstracts_index_0_1'
map     = "pubmed_abstracts_mapping_0_1"
es      = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)


bod     = {
    "query": {
        "bool": {
            "must": [
                {
                    "range" : {
                        "DateCreated" : {
                            "gte": "2000",
                            "lte": "2018",
                            "format": "dd/MM/yyyy||yyyy"
                        }
                    }
                },
                {
                    "query_string": {
                        "query": "What is the treatment of choice  for gastric lymphoma"
                    }
                }
            ]
        }
    }
}

res = es.search(index=index, doc_type=map, body=bod)


for item in res:
    pprint(item)
    exit()
