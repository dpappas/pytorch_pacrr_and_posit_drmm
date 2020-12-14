
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
    # doc_type:{
        "properties":{
            ######### TEXT
            'joint_text'  : {
                "type"          : "text",
                "analyzer"      : "english",
                "similarity"    : "my_similarity",
            },
            ######### KEYWORDS
            'pmcid'             : {"type": "keyword"},
            'doi'               : {"type": "keyword"},
            'pmid'              : {"type": "keyword"},
            ######### DATES
            'date'       : {
                "type"      : "date",
                "format"    : "yyyy-MM-dd"
            },
            ######### NESTED
            "section"          : {
                "type"      : "text",
                "analyzer"  : "english",
                "fields": {
                    "raw": {
                        "type": "keyword"
                    }
                }
            },
            ######### NESTED
            "doc_vec_scibert": {
                "type": "dense_vector",
                "dims": 768
            },
        }
    # }
    }
}

pprint(elastic_con.indices.create(index = index, ignore=400, body=mapping))
