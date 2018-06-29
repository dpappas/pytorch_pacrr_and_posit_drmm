
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from pprint import pprint

index   = 'pubmed_abstracts_index_0_1'
map     = "pubmed_abstracts_mapping_0_1"
es      = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)


query = {
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


for item in scan( es, query=query, index=index, doc_type=map):
    pprint(item)
    exit()
