
from elasticsearch import Elasticsearch
from pprint import pprint

index       = 'covid_index_0_1'
doc_type    = 'covid_mapping_0_1'
elastic_con = Elasticsearch(['127.0.01:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
elastic_con.indices.delete(index=index, ignore=[400,404])

mapping     = {
    "settings" : {
        "index" : {
            "similarity" : {
              "my_similarity" : {
                "type"  : "BM25",
                "k1"    : "0.6",
                "b"     : "0.3"
              }
            }
        }
    },
    "mappings":{
        "properties":{
            'joint_text'  : {
                "type"          : "text",
                "analyzer"      : "english",
                "similarity"    : "my_similarity",
            },
            'type'              : {"type": "keyword"},
        }
    # }
    }
}

pprint(elastic_con.indices.create(index = index, ignore=400, body=mapping))
