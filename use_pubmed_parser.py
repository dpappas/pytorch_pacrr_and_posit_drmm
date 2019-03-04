


import pubmed_parser as pp
from pprint import pprint

#############################
# path        = '/home/dpappas/for_ryan/pubmed19n0376.xml.gz'
# path        = '/home/dpappas/for_ryan/pubmed19n0403.xml.gz'
path        = '/home/dpappas/for_ryan/pubmed19n0437.xml.gz'
#############################
dicts_out   = pp.parse_medline_xml(path, year_info_only=False, nlm_category=False)
for item in dicts_out:
    if(len(item['abstract'].strip())>0):
        pprint(item)
        break
#############################














