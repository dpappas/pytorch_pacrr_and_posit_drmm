


import pubmed_parser as pp
from pprint import pprint
path        = '/home/dpappas/for_ryan/pubmed19n0437.xml.gz'
#############################
dicts_out   = pp.parse_medline_xml(path)
pprint(dicts_out[-1])
dicts_out   = pp.parse_medline_xml(path, year_info_only=False, nlm_category=False)
pprint(dicts_out[-1])
#############################














