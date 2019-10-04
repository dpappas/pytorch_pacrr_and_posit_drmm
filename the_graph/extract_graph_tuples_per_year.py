
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from collections import Counter
from tqdm import tqdm
from pprint import pprint
import pickle, sys, os

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
                names[ident].update(Counter([name]))
            else:
                names[ident] = Counter([name])
    ############################################################
    t = []
    for i in range(len(annots)-1):
        for j in range(i+1, len(annots)):
            id1, id2 = sorted([annots[i], annots[j]])
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

odir = '/home/dpappas/graph_data_pubtator/'
if(not os.path.exists(odir)):
    os.makedirs(odir)

index       = 'pubtator_annotations_0_1'
doc_type    = 'pubtator_annotations_map_0_1'

pmid_from   = int(sys.argv[1])
pmid_to     = int(sys.argv[2])

bod         = create_body(str(pmid_from), str(pmid_to))
items       = scan(es, query=bod, scroll='35m', index=index, doc_type=doc_type)
names       = {}
connections = Counter()
pbar        = tqdm(items, total=26320366)
for item in pbar:
    pmid = int(item['_source']['pmid'])
    if(pmid <= pmid_to and pmid >= pmid_from):
        pbar.set_description('{}||{}'.format(pmid, len(connections)))
        get_all_annots(item)

pickle.dump(names,       open(os.path.join(odir, 'graph_names_{}_{}.p'.format(pmid_from, pmid_to)), 'wb'))
pickle.dump(connections, open(os.path.join(odir, 'graph_connections_{}_{}.p'.format(pmid_from, pmid_to)), 'wb'))

'''

python3.6 extr_graph_data.py 0        1000000 &
python3.6 extr_graph_data.py 1000000  2000000 &
python3.6 extr_graph_data.py 2000000  3000000 &
python3.6 extr_graph_data.py 3000000  4000000 &
python3.6 extr_graph_data.py 4000000  5000000

python3.6 extr_graph_data.py 5000000  6000000 &
python3.6 extr_graph_data.py 6000000  7000000 &
python3.6 extr_graph_data.py 7000000  8000000 &
python3.6 extr_graph_data.py 8000000  9000000 &
python3.6 extr_graph_data.py 9000000  10000000
python3.6 extr_graph_data.py 10000000 11000000 &
python3.6 extr_graph_data.py 11000000 12000000 &
python3.6 extr_graph_data.py 12000000 13000000 &
python3.6 extr_graph_data.py 13000000 14000000 &
python3.6 extr_graph_data.py 14000000 15000000
python3.6 extr_graph_data.py 15000000 16000000 &
python3.6 extr_graph_data.py 16000000 17000000 &
python3.6 extr_graph_data.py 17000000 18000000 &
python3.6 extr_graph_data.py 18000000 19000000 &
python3.6 extr_graph_data.py 19000000 20000000 &
python3.6 extr_graph_data.py 20000000 21000000 &
python3.6 extr_graph_data.py 21000000 22000000 
python3.6 extr_graph_data.py 22000000 23000000 &
python3.6 extr_graph_data.py 23000000 24000000 &
python3.6 extr_graph_data.py 24000000 25000000 &
python3.6 extr_graph_data.py 25000000 26000000 &
python3.6 extr_graph_data.py 26000000 27000000 &
python3.6 extr_graph_data.py 27000000 28000000 &
python3.6 extr_graph_data.py 28000000 29000000


import pickle
from pprint import pprint
connections = pickle.load(open('/home/dpappas/graph_data_pubtator/graph_connections_8000000_9000000.p', 'rb'))
names       = pickle.load(open('/home/dpappas/graph_data_pubtator/graph_names_8000000_9000000.p', 'rb'))
pprint(connections.most_common(200000)[-100:])

for item in connections.most_common(480000)[-10:]:
    pprint((item[0][0], names[item[0][0]]))

for item in connections:
    if(item[0] == 'MESH:D004617' and connections[item]>1):
        print((list(names[item[0]].keys())[0], list(names[item[1]].keys())[0], connections[item]))

pprint(names['925'])
pprint(names['MESH:D054066'])

# ps -ef | grep "extr_graph_data.py" | awk '{print $3}' | xargs kill -9 $1


'''



