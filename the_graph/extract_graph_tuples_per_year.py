
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from collections import Counter
from tqdm import tqdm
from pprint import pprint
import pickle, sys

def create_body(pmid_from, pmid_to):
    return {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "pmid": {
                                "gte": pmid_from,
                                "lte": pmid_to
                            }
                        }
                    }
                ]
            }
        }
    }

def get_all_annots(item):
    ############################################################
    annots = []
    for p in item['_source']['passages']:
        for annot in p['annotations']:
            ident, name = annot['identifier'], annot['text']
            if(len(ident.strip())==0):
                # ident = name
                continue
            annots.append(ident)
            if(ident in names):
                names[ident].add(name)
            else:
                names[ident] = set([name])
    ############################################################
    t = []
    for i in range(len(annots)-1):
        for j in range(i+1, len(annots)):
            id1, id2 = annots[i], annots[j]
            t.append((id1, id2))
    ############################################################
    connections.update(Counter(t))

with open('elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if(len(line.strip())>0)]
    fp.close()

es = Elasticsearch(
    cluster_ips,
    verify_certs        = True,
    timeout             = 150,
    max_retries         = 10,
    retry_on_timeout    = True
)

index       = 'pubtator_annotations_0_1'
doc_type    = 'pubtator_annotations_map_0_1'

pmid_from   = int(sys.argv[1])
pmid_to     = int(sys.argv[2])

bod         = create_body(str(pmid_from), str(pmid_to))
items       = scan(es, query=bod, index=index, doc_type=doc_type)
names       = {}
connections = Counter()
pbar        = tqdm(items, total=26320366)
for item in pbar:
    pmid = int(item['_source']['pmid'])
    if(pmid <= pmid_to and pmid >= pmid_from):
        pbar.set_description('{}||{}'.format(pmid, len(connections)))
        get_all_annots(item)

pickle.dump(names,       open('graph_names_{}_{}.p'.format(pmid_from, pmid_to), 'wb'))
pickle.dump(connections, open('graph_connections_{}_{}.p'.format(pmid_from, pmid_to), 'wb'))

'''
python3.6 extr_graph_data.py 15000000 16000000 
python3.6 extr_graph_data.py 20000000 21000000 
python3.6 extr_graph_data.py 21000000 22000000 
python3.6 extr_graph_data.py 22000000 23000000 
python3.6 extr_graph_data.py 23000000 24000000 
python3.6 extr_graph_data.py 24000000 25000000
python3.6 extr_graph_data.py 25000000 26000000

'''



