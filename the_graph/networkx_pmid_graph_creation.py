

import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path

fpaths = [
    'C:\\Users\\dvpap\\Downloads\\citations_2016_2018.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_1900_1989.txt'
]

G   = nx.Graph()

for fpath in fpaths:
    for l in open(fpath):
        from_, to_ = l.strip().split(':')
        from_, to_ = from_.strip(), to_.strip()
        if len(to_.strip() == 0):
            continue
        print(from_, to_)
        G.add_edge(from_, to_)

print(shortest_path(G, source='5335695', target='2503568', weight=None, method="dijkstra"))
