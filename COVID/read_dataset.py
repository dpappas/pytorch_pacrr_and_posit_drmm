

from tqdm import tqdm
from pprint import pprint
import tarfile, json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def create_an_action(elk_dato, id):
    elk_dato['_id']      = id
    elk_dato['_op_type'] = 'index'
    elk_dato['_index']   = index
    elk_dato['_type']    = doc_type
    return elk_dato

def upload_to_elk():
    global actions
    flag = True
    while (flag):
        try:
            result = bulk(es, iter(actions))
            pprint(result)
            flag = False
        except Exception as e:
            if ('ConnectionTimeout' in str(e) or 'rejected execution of' in str(e)):
                print('Retrying')
            else:
                print(e)
                flag = False
    actions         = []

index       = 'allenai_covid_index_2020_11_29'
doc_type    = 'allenai_covid_mapping_2020_11_29'

es          = Elasticsearch(['127.0.0.1:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

targz_path  = "/media/dpappas/dpappas_data/CORD_allenai_datasets/2020-11-29/document_parses.tar.gz"
tar         = tarfile.open(targz_path, "r:gz")

total_items         = 0
total_paragraphs    = 0
database_instances  = []
actions             = []
for member in tqdm(tar, total=234501):
    # print(member.name)
    total_items += 1
    f = tar.extractfile(member)
    if f is not None:
        c = 0
        d = json.loads(f.read())
        title       = d['metadata']['title']
        datum       = {
            '_id'   : d['paper_id']+ ' ' +str(c),
            'text'  : title,
            'type'  : 'title',
            'rank'  : c
        }
        actions.append(create_an_action(datum, datum['_id']))
        # upload_to_elk(finished=False)
        c += 1
        if 'abstract' in d:
            abstract    = '\n'.join([t['text'] for t in d['abstract']])
            datum       = {
                'id'    : d['paper_id']+ ' ' +str(c),
                'text'  : abstract,
                'type'  : 'abstract',
                'rank'  : c
            }
            actions.append(create_an_action(datum, datum['_id']))
            c += 1
        # lezantes    = '\n'.join([t['text'] for t in d['ref_entries'].values()])
        if('PMC' in member.name):
            for par in d['body_text']:
                lezantes = '\n'.join(
                    [
                        d['ref_entries'][ref_item['ref_id']]['text']
                        for ref_item in par['ref_spans']
                        if ref_item['ref_id']
                    ]
                )
                par_text = par['text'] + '\n\n' + lezantes
                par_text = par_text.strip()
                datum       = {
                    'id'    : d['paper_id']+ ' ' +str(c),
                    'text': par_text,
                    'type': 'paragraph',
                    'rank'  : c
                }
                actions.append(create_an_action(datum, datum['_id']))
                c += 1
        else:
            for par in d['body_text']:
                lezantes = '\n'.join(
                    [
                        d['ref_entries'][ref_item['ref_id']]['text']
                        for ref_item in par['ref_spans']
                        if ref_item['ref_id']
                    ]
                )
                par_text = par['text'] + '\n\n' + lezantes
                par_text = par_text.strip()
                datum       = {
                    'id'    : d['paper_id']+ ' ' +str(c),
                    'text'  : par_text,
                    'type'  : 'paragraph',
                    'rank'  : c
                }
                actions.append(create_an_action(datum, datum['_id']))
                c += 1
    upload_to_elk(finished=False)

upload_to_elk(finished=True)