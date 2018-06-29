
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from pprint import pprint

index       = 'pubmed_abstracts_index_0_1'
map         = "pubmed_abstracts_mapping_0_1"
es          = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)

items   = scan( es, query=None, index=index, doc_type=map)
m =  0
for item in items:
    if(
        len(item['_source']['AbstractText'].strip()) == 0
        or
        len(item['_source']['ArticleTitle'].strip()) == 0
    ):
        print(es.delete(index=index, doc_type=map, id=item['_id']))
    m += 1
    print m





