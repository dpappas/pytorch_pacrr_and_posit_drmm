
from elasticsearch import Elasticsearch
from pprint import pprint

index       = 'history_index_0_1'
doc_type    = 'history_mapping_0_1'
elastic_con = Elasticsearch(['127.0.01:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
elastic_con.indices.delete(index=index, ignore=[400,404])

mapping = {
    "mappings":{
        "properties":{
            ######### TEXT
            'qtext'  : {
                "type"          : "text",
                "analyzer"      : "english"
            },
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
            }
        }
    }
}

pprint(elastic_con.indices.create(index = index, ignore=400, body=mapping))
