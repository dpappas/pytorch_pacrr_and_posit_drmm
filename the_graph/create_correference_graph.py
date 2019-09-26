
import gzip
from lxml import etree
from xml.dom import minidom
from pprint import pprint
from tqdm import tqdm
from collections import Counter
import random
import os

flattened = lambda l: ['{}:::{}'.format(item['type'], item['text']) for sublist in l['passages'] for item in sublist['annotations']]

def xml2json():
    ret = {
        'pmid'      : ch_tree.find('id').text.strip(),
        'passages'  : []
    }
    for passage in ch_tree.findall('passage'):
        passage_text = passage.find('text').text
        if(passage_text is None):
            passage_text = ''
        else:
            passage_text = passage_text.strip()
        p = {
            'section'       : passage.find('infon').text.strip(),
            'offset'        : int(passage.find('offset').text.strip()),
            'text'          : passage_text,
            'annotations'   : []
        }
        for anot in passage.findall('annotation'):
            idd = [t for t in anot.findall('infon') if ('identifier' in t.attrib.values())]
            idd = idd[0].text.strip() if(len(idd)>0) else ''
            typ = [t for t in anot.findall('infon') if ('type' in t.attrib.values())]
            typ = typ[0].text.strip() if(len(typ)>0) else ''
            medic   = [t for t in anot.findall('infon') if ('MEDIC' in t.attrib.values())]
            medic   = medic[0].text.strip() if(len(medic)>0) else ''
            mesh    = [t for t in anot.findall('infon') if ('MESH' in t.attrib.values())]
            mesh    = mesh[0].text.strip() if(len(mesh)>0) else ''
            ncbi_gene   = [t for t in anot.findall('infon') if ('NCBI Gene' in t.attrib.values())]
            ncbi_gene   = ncbi_gene[0].text.strip() if(len(ncbi_gene)>0) else ''
            ncbi_tax    = [t for t in anot.findall('infon') if ('NCBI Taxonomy' in t.attrib.values())]
            ncbi_tax    = ncbi_tax[0].text.strip() if(len(ncbi_tax)>0) else ''
            p['annotations'].append(
                {
                    'offset'        : int(anot.find('location').get('offset').strip()),
                    'length'        : int(anot.find('location').get('length').strip()),
                    'text'          : anot.find('text').text.strip(),
                    'identifier'    : idd,
                    'type'          : typ,
                    'medic'         : medic,
                    'mesh'          : mesh,
                    'ncbi_gene'     : ncbi_gene,
                    'ncbi_taxonomy' : ncbi_tax
                }
            )
        ret['passages'].append(p)
    return ret

def pretty_print_xml(ch_tree):
    the_tag         = ch_tree.tag
    rough_string    = etree.tostring(ch_tree, encoding='utf-8')
    reparsed        = minidom.parseString(rough_string)
    print(reparsed.toprettyxml(indent="\t"))

def create_pmid_body(pmid):
    return {
        "query": {
            "constant_score": {
                "filter": {
                    'bool': {
                        "must": [
                            {
                                "term": {
                                    "pmid": pmid
                                }
                            },
                        ]
                    }
                }
            }
        }
    }

diri        = '/media/dpappas/dpappas_data/PUBTATOR/'
paths       = [os.path.join(diri,f) for f in os.listdir(diri)]
random.shuffle(paths)
b_size      = 1000
actions     = []

count       = Counter()
coocurences = []
for path in tqdm(paths):
    root = etree.iterparse(gzip.open(path, 'rb'), tag='document')
    metr    = 0
    for _, ch_tree in tqdm(root, total=1000000):
        metr        += 1
        datum       = xml2json()
        terms       = flattened(datum)
        coocurences.append(terms)
        count.update(Counter(terms))




