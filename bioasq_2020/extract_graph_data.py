


# python3.6
import pickle
from pprint import pprint

names 		= pickle.load(open('/media/dpappas/dpappas_data/graph_data_pubtator/graph_names_27000000_28000000.p', 'rb'))
connections = pickle.load(open('/media/dpappas/dpappas_data/graph_data_pubtator/graph_connections_27000000_28000000.p', 'rb'))

for ((id1, id2), cc) in sorted(connections.items(), key=lambda x: x[1], reverse=True):
    if(id1 == id2):
        continue
    break

pprint(next(iter(names.items())))
pprint(next(iter(connections.items())))

# graph_names_7000000_8000000.p
# graph_names_8000000_9000000.p

pprint(names['9971'])
pprint((names['MESH:D009369']))















