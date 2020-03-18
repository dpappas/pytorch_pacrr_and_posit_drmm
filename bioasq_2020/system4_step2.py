
from elasticsearch import Elasticsearch
from pprint import pprint
import json, os
from tqdm import tqdm

b                       = '2'
odir                    = '/home/dpappas/bioasq_2020/system4_output_b{}/'.format(b)
bioasq_test_set_fpath   = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_{}/BioASQ-task8bPhaseA-testset{}'.format(b, b)

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

dd  = json.load(open(os.path.join(odir, 'bioasq_results.json')))

with open('/home/dpappas/elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if(len(line.strip())>0)]
    fp.close()

es = Elasticsearch(cluster_ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

# index, doc_type = 'pubmed_abstracts_0_1', 'abstract_map_0_1'
index, doc_type = 'pubmed_abstracts_joint_0_1', 'abstract_map_joint_0_1'

qdata                   = json.load(open(bioasq_test_set_fpath))
qid2text                = dict([(q['id'], q['body']) for q in qdata['questions']])

all_subm_data       = {'questions': []}
pbar                = tqdm(dd)
for qid in pbar:
    q_subm_data = {
        "id"        : qid,
        "body"      : qid2text[qid],
        "documents" : [],
        "snippets"  : []
    }
    ########################################################################
    exclude_pmids   = set()
    counter         = 10
    for snip_score, _, snip in dd[qid]:
        pmid, _, snip = snip.split(':',2)
        if(snip == 'GSK3732394 is currently in clinical trials.'):
            snip = 'GSK3732394 is currently in human studies.'
        pbar.set_description(pmid)
        try:
            doc_data    = es.get(index, doc_type, pmid)
        except:
            exclude_pmids.add(pmid)
            continue
        ########################################################################
        # if(doc_data['_source']['AbstractText'] is None or len(doc_data['_source']['AbstractText'])==0):
        #     continue
        # title           = doc_data['_source']['ArticleTitle'].strip()
        # abstract        = doc_data['_source']['AbstractText'].strip()
        title           = doc_data['_source']['joint_text'].split('--------------------')[0].strip()
        abstract        = doc_data['_source']['joint_text'].split('--------------------')[1].strip()
        ########################################################################
        title           = ' '.join(
            [
                tok for tok in title.split()
                if(not (tok.startswith('__') and tok.endswith('__')))
            ]
        )
        abstract        = ' '.join(
            [
                tok for tok in abstract.split()
                if(not (tok.startswith('__') and tok.endswith('__')))
            ]
        )
        snip            = ' '.join(
            [
                tok for tok in snip.split()
                if(not (tok.startswith('__') and tok.endswith('__')))
            ]
        )
        ########################################################################
        if(snip.startswith('[')):
            snip = snip[1:]
        if(snip.endswith('.')):
            snip = snip[:-1]
        if(snip.endswith(']')):
            snip = snip[:-1]
        snip = snip.strip()
        ########################################################################
        pbar.set_description('|{}: {}|'.format(pmid, snip))
        ########################################################################
        try:
            ind_from    = title.lower().index(snip.lower())
            ind_to      = ind_from + len(snip)
            section     = 'title'
        except:
            ind_from    = abstract.lower().index(snip.lower())
            ind_to      = ind_from + len(snip)
            section     = 'abstract'
        q_subm_data["snippets"].append(
            {
                "beginSection"          : section,
                "endSection"            : section,
                "document"              : "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pmid),
                "offsetInBeginSection"  : ind_from,
                "offsetInEndSection"    : ind_to,
                "text"                  : snip
            }
        )
        ########################################################################
        counter -= 1
        if(counter == 0):
            break
    ########################################################################
    doc_ids = f7([item[2].split(':')[0].strip() for item in dd[qid]])
    doc_ids = [doc_id for doc_id in doc_ids if doc_id not in exclude_pmids][:10]
    q_subm_data["documents"] = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(doc_id) for doc_id in doc_ids]
    ########################################################################
    all_subm_data['questions'].append(q_subm_data)

if (not os.path.exists(odir)):
    os.makedirs(odir)

with open(os.path.join(odir, 'bioasq8_batch{}_system4_results.json'.format(b)), 'w') as f:
    gb = f.write(json.dumps(all_subm_data, indent=4, sort_keys=True))
