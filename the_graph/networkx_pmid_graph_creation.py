

import networkx as nx
import numpy as np
from tqdm import tqdm
from networkx.algorithms.shortest_paths.generic import shortest_path

fpaths = [
    'C:\\Users\\dvpap\\Downloads\\citations_1900_1989.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_1990_1999.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2000_2004.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2005_2009.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2010_2015.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2016_2018.txt',
    'C:\\Users\\dvpap\\Downloads\\citations_2019_2021.txt',
    #
    'C:\\Users\\dvpap\\Downloads\\pmid2issn_2000_2004.txt',
    'C:\\Users\\dvpap\\Downloads\\pmid2issn_2005_2009.txt',
    'C:\\Users\\dvpap\\Downloads\\pmid2issn_2016_2018.txt',
    #'C:\\Users\\dvpap\\Downloads\\.txt',
]

G   = nx.Graph()

for fpath in fpaths:
    print(fpath)
    for l in tqdm(open(fpath)):
        # print((fpath, l.strip()))
        from_, to_ = l.strip().split(':', 1)
        from_, to_ = from_.strip(), to_.strip()
        if len(to_.strip()) == 0:
            continue
        G.add_node(from_)
        G.add_node(to_)
        G.add_edge(from_, to_)
        G.add_edge(to_, from_)

some_nodes = ['30587508', '24850885', '30047297', '31130811', '22621377', '21075122', '23870312', '29386389', '20237445', '22700207', '28122498', '29229824', '29146912', '24430003', '23527029', '25122230', '9760921', '21175058', '10483442', '23690948', '30074895', '29283327', '28950715', '25686628', '24524678', '12582577', '21480512', '30957403', '31622055', '32469335', '29226112', '23608633', '24987733', '17972643', '19449654', '29672266', '28499256', '8924901', '26981015', '21998654', '22713628', '27419175', '23816508', '18510017', '9720616', '26623746', '30774505', '20876996', '9869276', '27491753', '21659002', '21951598', '19493196', '27706109', '30736275', '29703496', '28009851', '22835152', '23226548', '31538676', '27087849', '24695515', '22022518', '23633469', '24273257', '32901512', '21112751', '21383140', '25410991', '29375639', '24178352', '30559756', '22983293', '31274421', '31271501', '27816677', '21698096', '24518360', '26945390', '28286698', '8892565', '26818000', '12587877', '27806047', '15975141', '7713947', '17141428', '11393615', '8099083', '32222657', '8489164', '12175947', '29479363', '3304661', '17912598', '12916667', '25472866', '27071261', '15074617', '1325443']
print(len(some_nodes))
some_nodes = [t for t in some_nodes if G.has_node(t)]
print(len(some_nodes))

# dists = np.zeros((len(some_nodes), len(some_nodes)))
# for i in range(len(some_nodes)):
#     for j in range(i, len(some_nodes)):
#         try:
#             dists[i, j] = shortest_path(G, source=some_nodes[i], target=some_nodes[j], weight=None, method="dijkstra")
#         except Exception as e:
#             print(e)
#             dists[i, j] = 100000
#
# print(dists)
#

