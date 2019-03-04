


import pubmed_parser as pp
from pprint import pprint
path        = '/home/dpappas/for_ryan/pubmed19n0437.xml.gz'
# dict_out    = pp.parse_pubmed_xml(path)
dicts_out   = pp.parse_medline_xml(path)
pprint(dicts_out[0])














