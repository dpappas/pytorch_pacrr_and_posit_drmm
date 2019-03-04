


import  os, sys, pymongo
import  pubmed_parser as pp
from    pprint import pprint
from    tqdm import tqdm

#############################
db_name             = 'pubmedBaseline2019'
collection_name     = 'articles'
client              = pymongo.MongoClient("localhost", 27017, maxPoolSize=50)
mongo_collection    = client[db_name][collection_name]
print(mongo_collection.count())
# mongo_collection.drop()
# mongo_collection.create_index("pmid", unique=True)
#############################
diri                = '/home/DATA/pubmed_baseline_2018/'
all_xml_gzs     = [os.path.join(diri, f) for f in os.listdir(diri) if f.endswith('.xml.gz')]
all_xml_gzs     = sorted(all_xml_gzs)
#############################
file_from           = int(sys.argv[1])
file_to             = int(sys.argv[2]) # 971
for path in tqdm(all_xml_gzs[file_from:file_to]):
    dicts_out   = pp.parse_medline_xml(path, year_info_only=False, nlm_category=False)
    for item in tqdm(dicts_out):
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
            if(mongo_collection.find_one({'pmid': item['pmid']}) is None):
                res = mongo_collection.insert_one(mongo_datum)
            # pprint(res)
#############################

'''

vim index_xml_gz.py

python3.6 index_xml_gz.py 0 200
python3.6 index_xml_gz.py 200 400
python3.6 index_xml_gz.py 400 600
python3.6 index_xml_gz.py 600 800
python3.6 index_xml_gz.py 800 1000

#############################

# pprint(client['pubmedBaseline2018']['articles'].find_one())

#############################

u'abstractText'
u'journalName'
u'keywords'
u'meshHeadingsList'
u'pmid'
u'title'
u'publicationDate'

u'publicationTypeList'
'''











