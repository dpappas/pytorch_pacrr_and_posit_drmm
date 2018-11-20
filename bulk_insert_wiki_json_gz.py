
import json
import subprocess
from pprint import pprint
import random
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch import helpers

def abs_found(pmid):
    bod = {
        "query" : {
            'bool':{
                "must" : [
                    {
                        "term": {
                            "pmid": pmid
                        }
                    }
                ]
            }
        }
    }
    res = es.search(index=index, doc_type=doc_type, body=bod)
    return len(res['hits']['hits'])>0

def create_an_action(tw):
    tw['_op_type']= 'index'
    tw['_index']= index
    tw['_type']= doc_type
    return tw

def send_to_elk(actions):
    flag = True
    while (flag):
        try:
            result = bulk(es, iter(actions))
            pprint(result)
            flag = False
        except Exception as e:
            print(e)
            if ('ConnectionTimeout' in str(e)):
                print('Retrying')
            else:
                flag = False

es          = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
index       = 'wikipedia_json_gz'
doc_type    = "wiki_page"
b_size      = 200
actions     = []
fpath       = '/media/dpappas/dpappas_data/enwiki-20181112-cirrussearch-content.json.gz'
proc        = subprocess.Popen(["zcat",fpath], stdout=subprocess.PIPE)

for line in iter(proc.stdout.readline,''):
    line    = line.rstrip()
    dato    = json.loads(line)
    # pprint(dato)
    temp    = create_an_action(dato)
    actions.append(temp)
    if(len(actions) >= b_size):
        send_to_elk(actions)
        actions = []

if(len(actions) > 0):
    send_to_elk(actions)
    actions = []


'''
curl -X GET "harvester2.ilsp.gr:9200/_search" -H 'Content-Type: application/json' -d'
{
    "query": {
        "match_phrase" : {
            "text" : "Torque is the rotation equivalent"
        }
    },
}
'


GET wikipedia_json_gz/_search
{
  "_source": [ "title", "opening_text"],
  "query": {
    "bool": {
      "should": [
        {
          "multi_match" : {
            "query"                 : "who is Will Smith",
            "type"                  : "cross_fields",
            "fields"                : [ "text" ],
            "minimum_should_match"  : "50%" ,
            "slop"                  : 5
          }
        }
      ]
    }
  }
}



'''


