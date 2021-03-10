
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from collections import Counter
from tqdm import tqdm
from pprint import pprint
import pickle, sys, os

with open('/home/dpappas/elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if(len(line.strip())>0)]
    fp.close()

es = Elasticsearch(
    cluster_ips,
    verify_certs        = True,
    timeout             = 150,
    max_retries         = 10,
    retry_on_timeout    = True
)

year_from   = sys.argv[1] # 2010
year_to     = sys.argv[2] # 2015

bod         = {
    "query": {
        "bool": {
          "must": [
            {"query_string": {"query": "_exists_:references", "analyze_wildcard": True}},
            {"range": {"DateCompleted": {"gte": year_from, "lte": year_to, "format": "yyyy"}}}
          ]
        }
    }
}
items       = scan(es, query=bod, scroll='35m', index='pubmed_abstracts_0_1')
tot         = es.count(index= 'pubmed_abstracts_0_1', body=bod)['count']
names       = {}
connections = Counter()
pbar        = tqdm(items, total=tot)

with open('pmid2issn_{}_{}.txt'.format(year_from, year_to),'w') as fp:
    for item in pbar:
        if('references' in item['_source']):
            for ref in item['_source']['references']:
                fp.write('{} : {}\n'.format(item['_id'], ref['PMID']))
    fp.close()


'''
source /home/dpappas/venvs/elasticsearch_old/bin/activate
# python issn_pubmed.py 1000 1899
# python issn_pubmed.py 1900 1989
# python issn_pubmed.py 1990 1999
# python issn_pubmed.py 2000 2004
# python issn_pubmed.py 2005 2009
# python issn_pubmed.py 2010 2015
# python issn_pubmed.py 2016 2018
# python issn_pubmed.py 2019 2021
'''


