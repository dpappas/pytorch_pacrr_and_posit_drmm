

# python3.6
import pickle
from pprint import pprint
from tqdm import tqdm
from collections import Counter

all_names       = None
all_connections = None
for i in tqdm(range(23, 28)):
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

print(len(all_names))
print(len(all_connections))

at_least        = 10
all_connections = dict(item for item in all_connections.items() if(item[1]>=at_least))

print(len(all_connections))

def get_no_name(id1):
    name1 = all_names[id1].most_common(1)[0][0]
    try:
        no = name2no[name1]
    except:
        no = len(name2no)
        name2no[name1] = no
    return no, name1


name2no = {}
for ((id1, id2), cc) in tqdm(all_connections.items()):
    if(id1 == id2):
        continue
    no1, name1 = get_no_name(id1)
    no2, name2 = get_no_name(id2)


# pprint(next(iter(names.items())))
# pprint(next(iter(connections.items())))
#
# # graph_names_7000000_8000000.p
# # graph_names_8000000_9000000.p
#
# pprint(names['9971'])
# pprint((names['MESH:D009369']))















