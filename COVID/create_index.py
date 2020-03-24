
from elasticsearch import Elasticsearch

index   = 'covid_index_0_1'
map     = 'covid_map_0_1'

elastic_con   = Elasticsearch(['127.0.01:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
elastic_con.indices.delete(index=index, ignore=[400,404])

mapping = {
    "settings" : {
        "index" : {
            "similarity" : {
              "my_similarity" : {
                "type"  : "BM25",
                "k1"    : "1.2",
                "b"     : "0.75"
              }
            }
        }
    },
    "mappings":{
        map:{
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
                    "format"    : "yyy-MM-dd HH:mm:ss||yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||yyyy/MM/dd||dd/MM/yyyy||epoch_millis||EE MMM d HH:mm:ss Z yyyy"
                }
            }
        }
    }
}

print(elastic_con.indices.create(index = index, ignore=400, body=mapping))


'''

b   : a weight for doc length           default 0.75
k1  : a weight for term frequencies     default 1.2

curl -XPOST 'http://192.168.188.79:9201/pubmed_abstracts_joint_0_1/_close'
curl -XPUT "http://192.168.188.79:9201/pubmed_abstracts_joint_0_1/_settings" -d '
{
    "similarity": {
        "my_similarity": { 
            "type": "BM25",
            "b"  : 0.1,
            "k1" : 0.9
        }
    }
}'
curl -XPOST 'http://192.168.188.79:9201/pubmed_abstracts_joint_0_1/_open'




curl -XPUT "http://192.168.188.79:9201/pubmed_abstracts_joint_0_1" -d '
{
    "settings": {
        "similarity": {
            "my_similarity": { 
                "type": "BM25",
                "b"  : 0.1,
                "k1" : 0.9
            }
        }
    }
}
'

curl -XPUT "http://<server>/<index>" -d '
{
  "settings": {
    "similarity": {
      "custom_bm25": { 
        "type": "BM25",
        "b":    0 ,
         "k1" : 0.9
      }
    }
  }'
'''

