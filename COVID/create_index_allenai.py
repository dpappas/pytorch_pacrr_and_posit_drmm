
from elasticsearch import Elasticsearch
from pprint import pprint

index       = 'allenai_covid_index_2020_11_29_01'
doc_type    = 'allenai_covid_mapping_2020_11_29_01'

elastic_con = Elasticsearch(['127.0.0.1:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
elastic_con.indices.delete(index=index, ignore=[400,404])

mapping     = {
    "settings": {
     "index": {
      "similarity": {
        "my_similarity": {
            "type"              : "BM25",
            "k1"                : 0.6,
            "b"                 : 0.3,
            'discount_overlaps' : True
        }
      }
     }
    },
"   mappings":{
        "properties":{
            'section_text'  : {
                "type"              : "text",
                "analyzer"          : "standard",
                "similarity"        : "my_similarity",
            },
            'section_type'      : {"type": "keyword"},
        }
    }
}

pprint(elastic_con.indices.create(index = index, ignore=400, body=mapping))
