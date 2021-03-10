

import networkx as nx
from tqdm import tqdm
from networkx.algorithms.shortest_paths.generic import shortest_path

fpaths = [
    'C:\\Users\\dvpap\\Downloads\\citations_1900_1989.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_1990_1999.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2000_2004.txt',
    # '',
    'C:\\Users\\dvpap\\Downloads\\citations_2010_2015.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2016_2018.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2019_2021.txt'
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

some_nodes = ['27806047', '15975141', '7713947', '17141428']

import numpy as np

dists = np.zeros((len(some_nodes), len(some_nodes)))
for i in range(len(some_nodes)):
    for j in range(i, len(some_nodes)):
        try:
            dists[i, j] = shortest_path(G, source=some_nodes[i], target=some_nodes[j], weight=None, method="dijkstra")
        except Exception as e:
            print(e)
            dists[i, j] = 100000

print(dists)


