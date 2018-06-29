
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from pprint import pprint
from nltk.corpus import stopwords
import re

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

index   = 'pubmed_abstracts_index_0_1'
map     = "pubmed_abstracts_mapping_0_1"
es      = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)

stopWords   = set(stopwords.words('english'))
search_text = "What is the treatment of choice  for gastric lymphoma?"
search_text = ' '.join([ token for token in bioclean(search_text) if(token not in stopWords)])
print(search_text)

bod     = {
    "query": {
        "bool": {
            "must": [
                {
                    "range" : {
                        "DateCreated" : {
                            "gte": "2000",
                            "lte": "2018",
                            "format": "dd/MM/yyyy||yyyy"
                        }
                    }
                },
                {
                    "query_string": {
                        "query": search_text
                    }
                }
            ]
        }
    }
}

res = es.search(index=index, doc_type=map, body=bod)


for item in res['hits']['hits']:
    pprint(item)
    exit()
