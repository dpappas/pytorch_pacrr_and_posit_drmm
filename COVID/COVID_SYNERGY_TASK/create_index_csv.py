
from elasticsearch import Elasticsearch
from pprint import pprint

# index       = 'allenai_covid_index_2020_11_29_csv'
# index       = 'allenai_covid_index_2021_01_10_csv'
index       = 'allenai_covid_index_2021_01_25_csv'

elastic_con = Elasticsearch(['127.0.0.1:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
elastic_con.indices.delete(index='allenai_covid_index_2021_01_10_csv', ignore=[400,404])
elastic_con.indices.delete(index='allenai_covid_index_2020_11_29_csv', ignore=[400,404])
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
    "mappings":{
        "properties":{
            'joint_text'    : {
                "type"              : "text",
                "analyzer"          : "standard",
                "similarity"        : "my_similarity",
            },
            'cord_uid'      : {"type": "keyword"},
            'doi'           : {"type": "keyword"},
            'pubmed_id'     : {"type": "keyword"},
            'url'           : {"type": "keyword"},
            'publish_time'       : {
                "type"      : "date",
                "format"    : "yyyy-MM-dd||yyyy"
            },
        }
    }
}

pprint(elastic_con.indices.create(index = index, ignore=400, body=mapping))
