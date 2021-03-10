
import os, pickle
import networkx as nx
import numpy as np
from tqdm import tqdm
from pprint import pprint
from networkx.algorithms.shortest_paths.generic import shortest_path

# dir_path = 'C:\\Users\\dvpap\\Downloads\\'
dir_path = '/home/dpappas/'

fpaths = [
    'citations_1900_1989.txt',
    'citations_1990_1999.txt',
    'citations_2000_2004.txt',
    'citations_2005_2009.txt',
    'citations_2010_2015.txt',
    'citations_2016_2018.txt',
    'citations_2019_2021.txt',
    #
    'pmid2issn_1900_1989.txt',
    'pmid2issn_1990_1999.txt',
    'pmid2issn_2000_2004.txt',
    'pmid2issn_2005_2009.txt',
    'pmid2issn_2010_2015.txt',
    'pmid2issn_2016_2018.txt',
    'pmid2issn_2019_2021.txt',
]

save_path = '/media/dpappas/pubmed_pmid_issn_graph.nx'

if os.path.exists(save_path):
    G = pickle.load(open(save_path, 'rb'))
else:
    G   = nx.Graph()
    for fpath in fpaths:
        print(fpath)
        for l in tqdm(open(os.path.join(dir_path, fpath))):
            # print((fpath, l.strip()))
            try:
                from_, to_ = l.strip().split(':', 1)
                from_, to_ = from_.strip(), to_.strip()
                if len(to_.strip()) == 0:
                    continue
                G.add_node(from_)
                G.add_node(to_)
                G.add_edge(from_, to_)
                G.add_edge(to_, from_)
            except Exception as ex:
                print(ex)
    pickle.dump(G, open(save_path, 'wb'))

some_nodes = ['1363217', '1430197', '7729014', '8365725', '9282797', '11228173', '11535573', '12545275', '12750988', '14705112', '14991347', '16731295', '16740526', '16886897', '18597679', '19151369', '20879882', '20932824', '21572417', '21712793', '21878110', '22305980', '22366783', '22495311', '22729223', '22805709', '22842232', '23009675', '23020937', '23383720', '23453690', '23594499', '23810382', '24044690', '24075313', '24174593', '24385578', '24398019', '24579881', '24678776', '25076844', '25356899', '25401298', '25484024', '25502941', '25672852', '25985141', '26019872', '26054435', '26114861', '26189493', '26235985', '26244500', '26302956', '26319231', '26362251', '26362943', '26483451', '26602202', '26716362', '26763878', '27009151', '27036065', '27047663', '27108798', '27108799', '27256762', '27331024', '27450679', '27693233', '27894357', '28017370', '28035506', '28087732', '28193117', '28808027', '28937030', '28942967', '28973408', '29099659', '29138090', '29422393', '29460957', '29570242', '29878067', '29969170', '30349109', '30500825', '31054517', '31104363', '31250358', '31301044', '31537998', '32013205', '32567228', '32667908', '33111320', '33186543', '33186545', '33606261']
print(len(some_nodes))
some_nodes = [t for t in some_nodes if G.has_node(t)]
print(len(some_nodes))

dists = np.zeros((len(some_nodes), len(some_nodes)))
for i in range(len(some_nodes)):
    for j in range(i, len(some_nodes)):
        try:
            dists[i, j] = len(shortest_path(G, source=some_nodes[i], target=some_nodes[j], weight=None, method="dijkstra"))
        except Exception as e:
            print(e)
            dists[i, j] = 100000

print(dists)
pprint(list(zip(some_nodes,dists[:,some_nodes.index("33186545")].tolist())))


