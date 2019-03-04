


import  os, pymongo
import  pubmed_parser as pp
from    pprint import pprint
from    tqdm import tqdm

client              = pymongo.MongoClient("localhost", 27017, maxPoolSize=50)
db_name             = 'pubmedBaseline2019'
collection_name     = 'articles'
mongo_collection    = client[db_name][collection_name]
mongo_collection.drop()
mongo_collection.create_index("pmid", unique=True)

diri            = '/home/dpappas/for_ryan/'
all_xml_gzs     = [os.path.join(diri, f) for f in os.listdir(diri) if f.endswith('.xml.gz')]
all_xml_gzs     = sorted(all_xml_gzs)
#############################
for path in tqdm(all_xml_gzs):
    dicts_out   = pp.parse_medline_xml(path, year_info_only=False, nlm_category=False)
    for item in dicts_out:
        if(len(item['abstract'].strip()) > 0 and len(item['title'].strip()) > 0):
            mongo_datum = {
                'abstractText'      : item['abstract'],
                'title'             : item['title'],
                'journalName'       : item['journal'],
                'keywords'          : item['keywords'],
                'meshHeadingsList'  : [mesh.strip() for mesh in item['mesh_terms'].split(';')],
                'pmid'              : item['pmid'],
                'country'           : item['country'],
                'author'            : item['author'],
                'publicationDate'   : item['pubdate'],
            }
            res = mongo_collection.insert_one(mongo_datum)
            pprint(res)
#############################

'''
# pprint(client['pubmedBaseline2018']['articles'].find_one())

u'abstractText'
u'journalName'
u'keywords'
u'meshHeadingsList'
u'pmid'
u'title'
u'publicationDate'

u'publicationTypeList'
'''











