

# python3.6
import pickle
from pprint import pprint
from tqdm import tqdm
from collections import Counter

all_names       = None
all_connections = None
for i in tqdm(range(18, 28)):
    fname = '/media/dpappas/dpappas_data/graph_data_pubtator/graph_names_{}000000_{}000000.p'.format(i, i+1)
    cname = '/media/dpappas/dpappas_data/graph_data_pubtator/graph_connections_{}000000_{}000000.p'.format(i, i+1)
    names       = pickle.load(open(fname, 'rb'))
    connections = pickle.load(open(cname, 'rb'))
    if(not all_names):
        all_names       = names
        all_connections = connections
    else:
        for name in names:
            if(name in all_names):
                all_names[name].update(names[name])
            else:
                all_names[name] = names[name]
        for k, v  in connections.items():
            if(k in all_connections):
                all_connections[k] += v
            else:
                all_connections[k] = v

for ((id1, id2), cc) in tqdm(sorted(all_connections.items(), key=lambda x: x[1], reverse=True)):
    if(id1 == id2):
        continue
    break

pprint(next(iter(names.items())))
pprint(next(iter(connections.items())))

# graph_names_7000000_8000000.p
# graph_names_8000000_9000000.p

pprint(names['9971'])
pprint((names['MESH:D009369']))















