

import networkx as nx
from tqdm import tqdm
from networkx.algorithms.shortest_paths.generic import shortest_path

fpaths = [
    'C:\\Users\\dvpap\\Downloads\\citations_1900_1989.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_1990_1999.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2010_2015.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2016_2018.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2019_2021.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2000_2004.txt'
]

G   = nx.Graph()

for fpath in fpaths:
    print(fpath)
    for l in tqdm(open(fpath)):
        # print((fpath, l.strip()))
        from_, to_ = l.strip().split(':')
        from_, to_ = from_.strip(), to_.strip()
        if len(to_.strip()) == 0:
            continue
        G.add_edge(from_, to_)

print(shortest_path(G, source='5335695', target='2503568', weight=None, method="dijkstra"))
