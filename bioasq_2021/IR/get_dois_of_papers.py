

from elasticsearch import Elasticsearch
import pickle
from tqdm import tqdm

def get_doi(pmid):
    res = es.get(index=doc_index, id=pmid, request_timeout=120)
    tt = [idd for idd in res['_source']['OtherIDs'] if idd['Source']=='doi']
    if(len(tt)>0):
        return tt[0]['id']
    else:
        return 'None'

with open('/home/dpappas/elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if(len(line.strip())>0)]
    fp.close()

es = Elasticsearch(cluster_ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

doc_index = 'pubmed_abstracts_0_1'

d = pickle.load(open('/home/dpappas/bioasq_2021/test_batch_1/bm25_top100/bioasq9_bm25_top100.test.pkl','rb'))

with open('/home/dpappas/bioasq_2021/test_batch_1/bm25_top100/doc_ids.txt', 'w') as fo:
    for q in tqdm(d['queries']):
        doc_ids     = [tt['doc_id'] for tt in q['retrieved_documents']]
        doc_dois    = [get_doi(pmid) for pmid in doc_ids]
        gb          = fo.write('\nqid - {}\n'.format(q['query_id']))
        for x,y in zip(doc_ids, doc_dois):
            gb = fo.write('{} : {}\n'.format(x,y))
        # gb          = fo.write('{} : {}\n'.format(q['query_id'], ' '.join(doc_dois)))
    fo.close()


# import pickle, requests
# from bs4 import BeautifulSoup
#
# d = pickle.load(open('/home/dpappas/bioasq_2021/test_batch_1/bm25_top100/bioasq9_bm25_top100.test.pkl','rb'))
#
# with open('/home/dpappas/bioasq_2021/test_batch_1/bm25_top100/doc_ids.txt', 'w') as fo:
#     for q in d['queries']:
#         doc_ids = [tt['doc_id'] for tt in q['retrieved_documents']]
#         url     = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids={}'.format(','.join(doc_ids))
#         ret     = requests.get(url)
#         soup     = BeautifulSoup(ret.text)
#         for item in soup.find_all('record'):
#             print((item.get('pmid'), item.get('doi')))
#         gb = fo.write('{} : {}\n'.format(q['query_id'], ' '.join(doc_ids)))
#     fo.close()
