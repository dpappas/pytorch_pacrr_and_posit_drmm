
import json
from pprint import pprint
import collections
from tqdm import tqdm
import sys
nested_dict     = lambda: collections.defaultdict(nested_dict)

from elasticsearch import Elasticsearch
index, doc_type = 'pubmed_abstracts_joint_0_1', 'abstract_map_joint_0_1'

es      = Elasticsearch(['palomar.ilsp.gr:9201'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

b       = '5'

d1      = json.load(open('C:\\Users\\dvpap\\OneDrive\\Desktop\\BIOASQ_2020\\batch{}_submit_files\\ir_results\\system1_output_b{}\\v3 test_data_for_revision.json'.format(b, b)))
d2      = json.load(open('C:\\Users\\dvpap\\OneDrive\\Desktop\\BIOASQ_2020\\batch{}_submit_files\\ir_results\\system2_output_b{}\\v3 test_data_for_revision.json'.format(b, b)))

d3      = json.load(open('C:\\Users\\dvpap\\OneDrive\\Desktop\\BIOASQ_2020\\batch{}_submit_files\\ir_results\\system1_output_b{}\\v3 test_emit_bioasq.json'.format(b, b)))
d4      = json.load(open('C:\\Users\\dvpap\\OneDrive\\Desktop\\BIOASQ_2020\\batch{}_submit_files\\ir_results\\system2_output_b{}\\v3 test_emit_bioasq.json'.format(b, b)))

opath   = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\BIOASQ_2020\\batch{}_submit_files\\ir_results\\system4_output_b{}\\ensembe_sys1_and sys2.json'.format(b, b)

pmidtext_to_other   = {}
qid2qtext           = {}
for item in d3['questions']+d4['questions']:
    qid2qtext[item['id']] = item['body']
    for snip in item['snippets']:
        pmidtext_to_other['{}:{}'.format(
            snip['document'].replace('http://www.ncbi.nlm.nih.gov/pubmed/','').strip(),
            snip['text'].strip()
        )] = snip

combined_data_docs  = nested_dict()
combined_data_sents = nested_dict()

for qid, data in tqdm(list(d1.items()) + list(d2.items())):
    # pprint(data)
    # exit()
    ####################################################################################
    for docid in data['doc_scores']:
        try:
            combined_data_docs[qid][docid].append(data['doc_scores'][docid])
        except:
            combined_data_docs[qid][docid] = [data['doc_scores'][docid]]
    ####################################################################################
    for docid, snippets in data['snippets'].items():
        for snip in snippets:
            try:
                combined_data_sents[qid][docid][':'.join([docid, snip[3]])].append(snip[1])
            except:
                combined_data_sents[qid][docid][':'.join([docid, snip[3]])] = [snip[1]]


def get_pmid2text(sn):
    try:
        return pmidtext_to_other[sn]
    except:
        pmid        = sn.split(':')[0]
        snip        = sn.split(':',1)[1]
        doc_data    = es.get(index, doc_type, pmid)
        # pprint(doc_data['_source'])
        title       = doc_data['_source']['joint_text'].split('--------------------')[0].strip()
        abs         = doc_data['_source']['joint_text'].split('--------------------')[1].strip()
        if(snip in title):
            sec = 'title'
            fr  = title.index(snip)
            to  = fr + len(snip)
        else:
            sec = 'abstract'
            fr  = abs.index(snip)
            to  = fr + len(snip)
        return {
            "beginSection"          : sec,
            "document"              : "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pmid),
            "endSection"            : sec,
            "offsetInBeginSection"  : fr,
            "offsetInEndSection"    : to,
            "text"                  : snip
        }

extract_data = {'questions' : []}
for qid, doc_dat in tqdm(combined_data_docs.items()):
    max1        = max([abs(t[0]) for t in doc_dat.values()])
    max2        = max([abs(t[1]) for t in doc_dat.values()])
    # We used sum in batch 4 of bioasq 2020
    # doc_ids     = [t[0] for t in sorted(doc_dat.items(), key= lambda x : (x[1][0]/max1)+(x[1][1]/max2), reverse=True)[:10]]
    # We used max in batch 5 of bioasq 2020
    doc_ids   = [t[0] for t in sorted(doc_dat.items(), key= lambda x : max(x[1]), reverse=True)[:10]]
    # doc_ids   = [t for t in sorted(doc_dat.items(), key= lambda x : sum(x[1]), reverse=True)[:10]]
    # doc_ids   = [t[0] for t in sorted(doc_dat.items(), key= lambda x : x[1][0]*x[1][1], reverse=True)[:10]]
    # pprint(doc_ids)
    all_snips   = []
    for docid in doc_ids:
        # all_snips.extend(sorted(combined_data_sents[qid][docid].items(), key=lambda x: x[1][0]*x[1][1],reverse = True)[:2])
        all_snips.extend(sorted(combined_data_sents[qid][docid].items(), key=lambda x: max(x[1]),reverse = True)[:2])
    # We used sum in batch 4 of bioasq 2020
    # all_snips = [t[0] for t in sorted(all_snips, key=lambda x: sum(x[1]), reverse=True)[:10]]
    # We used max in batch 5 of bioasq 2020
    all_snips = [t[0] for t in sorted(all_snips, key=lambda x: max(x[1]), reverse=True)[:10]]
    # pprint(all_snips)
    # extract_data[qid] = {
    #     'doc_ids'   : doc_ids,
    #     'snips'     : all_snips,
    # }
    quest_data = {
        'id'            : qid,
        'body'          : qid2qtext[qid],
        'documents'     : [	"http://www.ncbi.nlm.nih.gov/pubmed/{}".format(ddd) for ddd in doc_ids],
        'snippets'      : [
            get_pmid2text(sn) for sn in all_snips
        ]
    }
    extract_data['questions'].append(quest_data)

with open(opath, 'w') as f:
    f.write(json.dumps(extract_data, indent=4, sort_keys=False))
    f.close()



'''
python3.6 combine_1_and_2.py 5
'''






