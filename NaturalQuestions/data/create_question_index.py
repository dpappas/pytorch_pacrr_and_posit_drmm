
from elasticsearch import Elasticsearch
from pprint import pprint

index   = 'natural_questions_q_0_1'
# map     = 'natural_questions_q_map_0_1'
lang    = 'english'

elastic_con = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
elastic_con.indices.delete(index=index, ignore=[400,404])

mapping = {
    "settings": {"analysis": {"analyzer": {"default": {"type": "english"}}}},
    "mappings":{
        # map:{
            "properties": {
                'example_id'        : {"type": "keyword"},
                'dataset'           : {"type": "keyword"},
                'document_url'      : {"type": "keyword"},
                'document_title'    : {"type": "text", "analyzer": 'english', "fields": {"raw": {"type": "keyword"}}},
                'question'          : {"type": "text", "analyzer": 'english', "fields": {"raw": {"type": "keyword"}}},
                'long_answer'       : {"type": "text", "analyzer": 'english', "fields": {"raw": {"type": "keyword"}}},
                'short_answer'      : {"type": "text", "analyzer": 'english', "fields": {"raw": {"type": "keyword"}}}
            }
        # }
    }
}

pprint(elastic_con.indices.create(index=index, ignore=400, body=mapping))
