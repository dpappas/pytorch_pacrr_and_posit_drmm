
import json
import subprocess
from pprint import pprint
import random
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch import helpers
import re

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

# print(clean_the_damn_thing(dato['source_text']))

def clean_the_damn_thing(tt):
    tt = tt.replace('&nbsp;', ' ').replace('<sub>', '').replace('</sub>', '').replace('<blockquote>', '').replace('</blockquote>', '')
    ####################################################################
    tt = re.sub(r'\[\[File:.*?\]\]',        r'',    tt, flags=re.DOTALL)
    tt = re.sub(r'\[\[Category:.*?\]\]',    r'',    tt, flags=re.DOTALL)
    # ####################################################################
    tt = re.sub(r'\[\[([^\|]+?)\]\]',       r'\1',  tt, flags=re.DOTALL)
    tt = re.sub(r'\[\[(.*?)\|(.*?)\]\]',    r'\2',  tt, flags=re.DOTALL)
    # ####################################################################
    tt = re.sub(r'<--.+?-->',               r'',    tt, flags=re.DOTALL)
    ####################################################################
    for i in range(4):
        for item in re.findall(r'<ref.*?</ref>', tt):
            if(item.count('<ref')==1):
                tt = tt.replace(item, '').strip()
        for item in re.findall(r'<ref.*?/>', tt):
            if(item.count('<ref')==1):
                tt = tt.replace(item, '').strip()
    ####################################################################
    for i in range(4):
        tt = re.sub(r'\{\{([^\{]+?)\}\}',   r'',    tt, flags=re.DOTALL)
    for i in range(4):
        tt = re.sub(r'\{\|[^\{]+?\|\}',     r'',    tt, flags=re.DOTALL)
    ####################################################################
    # tt = re.sub(r'==(.*?)==', '',  tt, flags=re.DOTALL)
    # ####################################################################
    for i in range(10):
        tt = tt.replace('\n\n\n', '\n\n')
    # tt = re.sub(r'<ref [^\<].+?/>',          r'',    tt, flags=re.DOTALL)
    # tt = re.sub(r'<ref.*?</ref>',         r'',    tt, flags=re.DOTALL)
    # re.findall(r'<ref [^\<].+?/>', clean_the_damn_thing(dato['source_text']))
    # re.findall(r'<ref.*?</ref>', clean_the_damn_thing(dato['source_text']))
    return tt.strip()

# pprint(re.findall(r'<ref .*?</ref>', clean_the_damn_thing(dato['source_text'])))

es          = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
index       = 'en_wikipedia_json_gz'
doc_type    = "wiki_page"
b_size      = 200
actions     = []
fpath       = '/media/dpappas/dpappas_data/enwiki-20181112-cirrussearch-content.json.gz'
# fpath       = '/media/dpappas/dpappas_data/elwiki-20190211-cirrussearch-content.json.gz'

proc        = subprocess.Popen(["zcat", fpath], stdout=subprocess.PIPE)
for line in iter(proc.stdout.readline, ''):
    line    = line.rstrip()
    dato    = json.loads(line)
    if('source_text' in dato):
        ##############################
        print(dato['title'])
        print(dato['timestamp'])
        print(dato['wikibase_item'])
        print(dato['category'])
        print(dato['create_timestamp'])
        ##############################
        # print(dato['opening_text'])
        paragraphs = re.split(r'\s\s+', dato['text'], flags=re.DOTALL)
        paragraphs = [
            par.strip()
            for par in paragraphs
            if(par is not None and len(par.strip()) > 0 and len(par.strip().split()) > 5)
        ]
        for par in paragraphs:
            print(par)
            print(5 * '-')
        print(20 * '=')
    # temp    = create_an_action(dato)
    # actions.append(temp)
    # if(len(actions) >= b_size):
    #     send_to_elk(actions)
    #     actions = []

if(len(actions) > 0):
    send_to_elk(actions)
    actions = []

'''

proc        = subprocess.Popen(["zcat", fpath], stdout=subprocess.PIPE)
for line in iter(proc.stdout.readline, ''):
    line    = line.rstrip()
    dato    = json.loads(line)
    if('source_text' in dato):
        print(dato['title'])
        clean_text = clean_the_damn_thing(dato['source_text'])
        paragraphs = []
        for p in re.split(r'===(.*?)===', clean_text, flags=re.DOTALL):
            for p2 in re.split(r'==(.*?)==', p, flags=re.DOTALL):
                for p3 in re.split(r'\s\s\s*', p2, flags=re.DOTALL):
                    if(p3 is not None and len(p3.strip()) > 0 and len(p3.strip().split()) > 5):
                        paragraphs.append(p3.strip())
        # paragraphs = re.split(r'\s\s+', clean_text)
        for par in paragraphs:
            print(par)
            print(5 * '-')
        print(20 * '=')
        # re.sub(r'\{\|.+?\|\}', r'', tt, re.DOTALL)
        # re.findall(r'\{\|.+?\|\}', tt, re.DOTALL)
        if('Battle of Diamond Rock'.lower() in dato['title'].lower()):
            break
    # temp    = create_an_action(dato)
    # actions.append(temp)
    # if(len(actions) >= b_size):
    #     send_to_elk(actions)
    #     actions = []

'''

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
            "fields"                : [ "text", "opening_text"],
            "minimum_should_match"  : "50%" ,
            "slop"                  : 2
          }
        }
      ]
    }
  }
}


'''


