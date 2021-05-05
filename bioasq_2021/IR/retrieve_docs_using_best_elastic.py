
import sys, os, re, json, pickle, ijson
from elasticsearch import Elasticsearch
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from pprint import pprint
import sys

similar_sets = [
set(['assessed', 'evaluated', 'analyzed', 'analysed', 'measured', 'examined']),
set(['found', 'observed']),
set(['database', 'databank']),
set(['genome', 'genome assembly']),
set(['xbp', 'xbps', 'atf']),
set(['coacervation', 'phase separation']),
set(['mitochondrion', 'mitochondria']),
set(['asians', 'caucasians', 'europeans']),
set(['breastfeeding', 'breast feeding', 'exclusive breastfeeding', 'childbirth']),
set(['activates', 'antagonizes']),
set(['mutation', 'missense mutation', 'pathogenic variant', 'nonsense mutation', 'gene mutation', 'point mutation', 'missense variant', 'deletion', 'frameshift mutation', 'novel mutation']),
set(['genes', 'gene families']),
set(['vaccine', 'influenza vaccine', 'dryvax', 'influenza vaccines', 'vaccines', 'recombinant vaccine']),
set(['monoclonal', 'polyclonal']),
set(['dickkopf', 'dkk']),
set(['expression', 'protein expression', 'expression level', 'mrna expression']),
set(['fibroids', 'myomas']),
set(['microduplication', 'microdeletion', 'q deletion', 'pathogenic variant']),
set(['alzheimers', 'alzheimer', 'huntingtons', 'alzheimer disease', 'alzheimers disease']),
set(['oncomine', 'gepia']),
set(['antibody', 'polyclonal', 'mab']),
set(['glioblastoma', 'gbm', 'glioma', 'glioblastoma multiforme', 'non small cell lung carcinoma', 'neuroblastoma', 'malignant gliomas', 'non small cell lung cancer nsclc', 'npc', 'triple negative breast cancer', 'sclc']),
set(['nanog', 'sox']),
set(['effect', 'effects']),
set(['abacavir', 'atazanavir', 'nevirapine']),
set(['endoderm', 'mesoderm', 'ectoderm', 'ureteric bud']),
set(['cyp', 'cyps', 'ugt', 'cytochrome p', 'cypa']),
set(['fingolimod', 'natalizumab', 'nintedanib']),
set(['pain', 'back pain']),
set(['promotes', 'accelerates', 'suppresses', 'restrains', 'augments']),
set(['located', 'situated']),
set(['rbioconductor', 'web tool', 'r package']),
set(['antagonists', 'agonists']),
set(['polymorphism', 'rs', 'genetic variant', 'rsag', 'genetic polymorphism']),
set(['pharmacogenetic', 'pharmacogenomic']),
set(['able', 'unable']),
set(['yeast', 'saccharomyces cerevisiae', 's cerevisiae']),
set(['used', 'utilized', 'employed', 'utilised', 'applied']),
set(['describe', 'discuss']),
set(['prevalence', 'prevalence rate']),
set(['efavirenz', 'stavudine', 'nevirapine', 'ritonavir', 'darunavir', 'rilpivirine', 'raltegravir', 'dolutegravir', 'tenofovir', 'zidovudine']),
set(['two', 'three', 'four']),
set(['trastuzumab', 'cetuximab', 'pertuzumab', 'erlotinib', 'panitumumab', 'afatinib', 'pembrolizumab', 'docetaxel', 'olaparib', 'lapatinib', 'atezolizumab']),
set(['autosomal', 'autosomal dominant', 'autosomal recessive']),
set(['azd', 'abt', 'panobinostat', 'chidamide', 'lonafarnib', 'icotinib', 'selinexor', 'gefitinib']),
set(['ofatumumab', 'alemtuzumab', 'ibrutinib', 'rituximab', 'denileukin diftitox', 'avelumab']),
set(['treatment', 'therapy']),
set(['ewings', 'ewing']),
set(['cypc', 'cypd', 'ugta', 'ugtb', 'vkorc']),
set(['positive', 'negative']),
set(['quantification', 'quantitation']),
set(['cgrp', 'galanin']),
set(['antibodies', 'igg antibodies', 'antisera']),
set(['effective', 'efficacious']),
set(['fda', 'us food and drug administration', 'us fda', 'european medicines agency', 'food and drug administration']),
set(['towards', 'toward']),
set(['propofol', 'remifentanil', 'dexmedetomidine', 'sevoflurane', 'desflurane', 'nitroglycerin', 'nicardipine', 'isoflurane', 'esmolol']),
set(['mellifera', 'apis mellifera']),
set(['recommendations', 'guidelines', 'clinical guidelines']),
set(['acupotomy', 'sgb']),
set(['molecules', 'ligands']),
set(['p', 'p≤']),
set(['clcn', 'kcnh', 'scna', 'atpa', 'kcnj']),
set(['proteins', 'polypeptides']),
set(['explain', 'reflect', 'accentuate', 'signify', 'downplay']),
set(['southern', 'northern', 'northeastern', 'southwestern', 'southeastern', 'northwestern', 'northeast', 'southwest', 'slovakia', 'southeast', 'north']),
set(['critical', 'crucial', 'important', 'essential', 'vital']),
set(['bacteria', 'microorganisms', 'microbes', 'fungi']),
set(['methodologies', 'approaches']),
set(['infection', 'salmonella infection']),
set(['zolmitriptan', 'solifenacin succinate']),
set(['hypofractionated', 'intensity modulated radiation therapy', 'whole breast', 'intensity modulated radiotherapy', 'normofractionated', 'hdr brachytherapy']),
set(['adalimumab', 'infliximab', 'etanercept', 'ustekinumab', 'golimumab', 'tcz', 'tocilizumab', 'vedolizumab', 'omalizumab', 'secukinumab', 'tofacitinib']),
set(['vesicles', 'membranes']),
set(['dexamethasone', 'dxm', 'dex']),
set(['cytokine', 'inflammatory cytokine']),
set(['domestic', 'feral']),
set(['package', 'spreadsheet']),
set(['aim', 'purpose', 'objective', 'objectives']),
set(['lncrnas', 'circrnas', 'mirnas', 'ncrnas', 'lincrnas', 'trna derived fragments', 'micrornas', 'mirs']),
set(['radiotherapy', 'radiation therapy', 'radiation treatment', 'external beam radiotherapy', 'external beam radiation therapy', 'palliative radiotherapy', 'concurrent chemoradiation', 'radiosurgery', 'chemoradiation', 'sbrt', 'whole brain radiotherapy']),
set(['september', 'july', 'august', 'october', 'march', 'november', 'february', 'june', 'april', 'december', 'january']),
set(['pseudotumor', 'pseudotumour']),
set(['pose', 'impose']),
set(['codeine', 'pseudoephedrine']),
set(['extracellular', 'intracellular']),
set(['children', 'adults', 'persons']),
set(['rhabdomyolysis', 'hyperammonemia', 'hyponatremia', 'kernicterus', 'hepatic encephalopathy']),
set(['hypercapnia', 'hypocapnia', 'hypercapnic']),
set(['color', 'colour']),
set(['polymorphisms', 'genetic polymorphisms', 'gene variants', 'single nucleotide polymorphisms']),
set(['among', 'amongst']),
set(['dipg', 'diffuse intrinsic pontine glioma']),
set(['foam', 'fabric']),
set(['sarcoma', 'rhabdomyosarcoma', 'lymphoma']),
set(['offers', 'provides', 'justifies', 'holds']),
set(['migraine', 'tension type headache', 'cluster headache', 'migraine with aura', 'migraines', 'migraine without aura']),
set(['rodents', 'nonhuman primates', 'humans']),
set(['recognized', 'recognised', 'regarded']),
set(['cyld', 'wwp', 'senp', 'skp', 'trim', 'eifa', 'smurf', 'famb']),
set(['variants', 'missense variants', 'novel variants', 'mutations']),
set(['kisspeptin', 'gnrh neurons', 'prrp']),
set(['basaloid', 'pleomorphic']),
set(['mother', 'father']),
set(['ligands', 'molecules']),
set(['panitumumab', 'pertuzumab', 'trastuzumab', 'ramucirumab', 'cetuximab']),
set(['toxoplasmosis', 'lyme disease', 'brucellosis', 'leptospirosis', 'strongyloidiasis']),
set(['repressed', 'regulated']),
set(['dkk', 'dickkopf', 'sfrp', 'activin a']),
set(['tumors', 'tumours', 'carcinomas', 'sarcomas', 'neoplasms']),
set(['regarding', 'concerning'])
]

def fix2(qtext):
    qtext = qtext.lower()
    if(qtext.startswith('can ')):
        qtext = qtext[4:]
    if(qtext.startswith('list the ')):
        qtext = qtext[9:]
    if(qtext.startswith('list ')):
        qtext = qtext[5:]
    if(qtext.startswith('describe the ')):
        qtext = qtext[13:]
    if(qtext.startswith('describe ')):
        qtext = qtext[9:]
    if('list as many ' in qtext and 'as possible' in qtext):
        qtext = qtext.replace('list as many ', '')
        qtext = qtext.replace('as possible', '')
    if('yes or no' in qtext):
        qtext = qtext.replace('yes or no', '')
    if('also known as' in qtext):
        qtext = qtext.replace('also known as', '')
    if('is used to ' in qtext):
        qtext = qtext.replace('is used to ', '')
    if('are used to ' in qtext):
        qtext = qtext.replace('are used to ', '')
    tokenized_body  = [t for t in qtext.split() if t not in stopwords]
    tokenized_body  = bioclean_mod(' '.join(tokenized_body))
    question        = ' '.join(tokenized_body)
    return question

def fix1(qtext):
    tokenized_body  = bioclean_mod(qtext)
    tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(tokenized_body)
    return question

# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod = lambda t: re.sub(
    '[.,?;*!%^&_+():-\[\]{}]', '',
    t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()
).split()
bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

doc_index = 'pubmed_abstracts_joint_0_1'

with open('/home/dpappas/elk_ips.txt') as fp:
    cluster_ips = [line.strip() for line in fp.readlines() if(len(line.strip())>0)]
    fp.close()

es = Elasticsearch(cluster_ips, verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

with open('/home/dpappas/bioasq_all/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

stopwords.add('with')
print(stopwords)

def tokenize(x):
  return bioclean(x)

def get_first_n_1(qtext, n, max_year=2022, expand=False):
    # tokenized_body  = bioclean_mod(qtext)
    # tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    # question        = ' '.join(tokenized_body)
    question = fix2(qtext)
    print(question)
    ################################################
    bod             = {
        "size": n,
        "query": {
            "bool": {
                "must": [{"range": {"DateCompleted": {"gte": "1900", "lte": str(max_year), "format": "dd/MM/yyyy||yyyy"}}}],
                "should": [
                    {
                        "match": {
                            "joint_text": {
                                "query": question,
                                "boost": 1,
                                'minimum_should_match': "30%"
                            }
                        }
                    },
                    {
                        "match": {
                            "joint_text": {
                                "query": question,
                                "boost": 1,
                                'minimum_should_match': "50%"
                            }
                        }
                    },
                    {
                        "match": {
                            "joint_text": {
                                "query": question,
                                "boost": 1,
                                'minimum_should_match': "70%"
                            }
                        }
                    },
                    {"match_phrase": {"joint_text": {"query": question, "boost": 1}}}
                ],
                "minimum_should_match": 1,
            }
        }
    }
    ################################################
    if expand:
        for token in set(question.split()):
            for similar_set in similar_sets:
                if token in similar_set:
                    bod['query']['bool']['should'].append(
                        {
                            "match": {
                                "joint_text": {
                                    "query": ' '.join(list(similar_set)),
                                    "boost": 1,
                                    'minimum_should_match': "{}%".format(int(100/len(similar_set)))
                                }
                            }
                        }
                    )
    ################################################
    res             = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

batch       = int(sys.argv[1])
# fpath       = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_{}/BioASQ-task8bPhaseA-testset{}'.format(batch,batch)
# odir        = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_{}/bioasq8_bm25_top100/'.format(batch)
# fpath       = '/home/dpappas/bioasq_2021/BioASQ-task9bPhaseA-testset{}'.format(batch,batch)
# odir        = '/home/dpappas/bioasq_2021/test_batch_{}/bm25_top100/'.format(batch)
fpath       = '/home/dpappas/bioasq_2021/BioASQ-task9bPhaseA-testset{}'.format(batch,batch)
odir        = '/home/dpappas/bioasq_2021/test_batch_{}/bm25_top100/'.format(batch)
test_data   = json.load(open(fpath))

test_docs_to_save = {}
test_data_to_save = {'queries' : []}

for q in tqdm(test_data['questions']):
    qtext       = q['body']
    print(qtext)
    qid         = q['id']
    #######################################################
    results     = get_first_n_1(qtext, 100, expand=True)
    print([t['_id'] for t in results])
    #######################################################
    temp_1      = {
        'num_rel'               : 0,
        'num_rel_ret'           : 0,
        'num_ret'               : -1,
        'query_id'              : qid,
        'query_text'            : qtext,
        'relevant_documents'    : [],
        'retrieved_documents'   : []
    }
    #######################################################
    all_scores          = [res['_score'] for res in results]
    # print(all_scores)
    scaler              = StandardScaler().fit(np.array(all_scores).reshape(-1,1))
    temp_1['num_ret']   = len(all_scores)
    #######################################################
    for res, rank in zip(results, range(1, len(results)+1)):
        test_docs_to_save[res['_id']] = {
            'abstractText'      : res['_source']['joint_text'].split('--------------------', 1)[1].strip(),
            'author'            : '',
            'country'           : '',
            'journalName'       : '',
            'keywords'          : '',
            'meshHeadingsList'  : [],
            'pmid'              : res['_id'],
            'publicationDate'   : res['_source']['DateCompleted'],
            'title'             : res['_source']['joint_text'].split('--------------------')[0].strip()
        }
        #######################################################
        temp_1['retrieved_documents'].append({
                'bm25_score'        : res['_score'],
                'doc_id'            : res['_id'],
                'is_relevant'       : False,
                'norm_bm25_score'   : scaler.transform([[res['_score']]])[0][0],
                'rank'              : rank
            })
    test_data_to_save['queries'].append(temp_1)
    # break

if(not os.path.exists(odir)):
    os.makedirs(odir)

pickle.dump(test_data_to_save, open(os.path.join(odir, 'bioasq9_bm25_top100.test.pkl'), 'wb'))
pickle.dump(test_docs_to_save, open(os.path.join(odir, 'bioasq9_bm25_docset_top100.test.pkl'), 'wb'))


'''
source /home/dpappas/venvs/elasticsearch_old/bin/activate
python retrieve_docs.py 5
'''


