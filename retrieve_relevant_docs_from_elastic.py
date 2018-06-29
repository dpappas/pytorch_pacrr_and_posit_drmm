
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from pprint import pprint
from nltk.corpus import stopwords
import json
import re

stopWords   = set(stopwords.words('english'))
index       = 'pubmed_abstracts_index_0_1'
map         = "pubmed_abstracts_mapping_0_1"
bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
es          = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)

def get_elk_results(search_text):
    search_text = ' '.join([token for token in bioclean(search_text) if (token not in stopWords)])
    print(search_text)
    bod = {
        'size': 1000,
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "DateCreated": {
                                "gte": "1900",
                                "lte": "2017",
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
    ret = {}
    for item in res['hits']['hits']:
        ret[item[u'_source']['pmid']] = item[u'_score']
    return ret


# bioasq_data_path    = '/home/DATA/Biomedical/bioasq6/bioasq6_data/BioASQ-trainingDataset6b.json'
bioasq_data_path    = '/home/dpappas/bioasq_ir_data/BioASQ-trainingDataset6b.json'
data                = json.load(open(bioasq_data_path, 'r'))
total               = len(data['questions'])
m                   = 0
for quest in data['questions']:
    qtext       = quest['body']
    pmids       = [d.split('/')[-1] for d in quest['documents']]
    print(qtext)
    print(pmids)
    print(min([float(f) for f in pmids]))
    print(max([float(f) for f in pmids]))
    elk_scored_pmids = get_elk_results(qtext)
    sorted_keys     = sorted(elk_scored_pmids.keys(), key=lambda x: elk_scored_pmids[x], reverse=True)
    my_truth_1000   = [ p in sorted_keys[:1000] for p in pmids ]
    my_truth_500    = [ p in sorted_keys[:500] for p in pmids  ]
    my_truth_100    = [ p in sorted_keys[:100] for p in pmids  ]
    my_truth_50     = [ p in sorted_keys[:50] for p in pmids   ]
    my_truth_10     = [ p in sorted_keys[:10] for p in pmids   ]
    my_truth_1000   = float(sum(my_truth_1000)) / float(len(my_truth_1000))
    my_truth_500    = float(sum(my_truth_500))  / float(len(my_truth_500))
    my_truth_100    = float(sum(my_truth_100))  / float(len(my_truth_100))
    my_truth_50     = float(sum(my_truth_50))   / float(len(my_truth_50))
    my_truth_10     = float(sum(my_truth_10))   / float(len(my_truth_10))
    print my_truth_10, my_truth_50, my_truth_100, my_truth_500, my_truth_1000
    m+=1
    print('Finished {} of {}'.format(m, total))
    print 20 * '-'

