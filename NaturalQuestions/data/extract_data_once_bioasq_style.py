
from elasticsearch  import Elasticsearch
from elasticsearch.helpers import scan
from tqdm import tqdm
from pprint import pprint
from bs4 import BeautifulSoup
import re, nltk
from difflib import SequenceMatcher
import pickle
import json
import os
import math

bioclean_mod    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()

stopwords       = nltk.corpus.stopwords.words("english")
elk_ip          = '192.168.188.80'

def clean_start_end(word):
    word = re.sub(r'(^\W+)', r'\1 ', word)
    word = re.sub(r'(\W+$)', r' \1', word)
    word = re.sub(r'\s+', ' ', word)
    return word.strip()

def tokenize(text):
    ret = []
    for token in nltk.tokenize.word_tokenize(text):
        ret.extend(clean_start_end(token).split())
    return ret

def get_first_n(question, n):
    question    = bioclean_mod(question)
    question    = [t for t in question if t not in stopwords]
    question    = ' '.join(question)
    ################################################
    doc_index   = 'natural_questions_0_1'
    es          = Elasticsearch(['{}:9200'.format(elk_ip)], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    ################################################
    bod         = {"size": n, "query": {"match": {"paragraph_text": question}}}
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# def get_first_n_11(question_tokens, n, idf_scores):
#     question    = ' '.join(question_tokens)
#     ################################################
#     the_shoulds = []
#     for q_tok, idf_score in zip(question_tokens, idf_scores):
#         the_shoulds.append({"match": {"paragraph_text": {"query": q_tok, "boost": idf_score}}})
#     if(len(question_tokens) > 1):
#         the_shoulds.append(
#             {
#                 "span_near": {
#                     "clauses": [{"span_term": {"paragraph_text": w}} for w in question_tokens],
#                     "slop": 5,
#                     "in_order": False
#                 }
#             }
#         )
#     ################################################
#     doc_index   = 'natural_questions_0_1'
#     es          = Elasticsearch(['{}:9200'.format(elk_ip)], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
#     ################################################
#     bod         = {
#         "size": n,
#         "query": {
#             "bool" : {
#                 "should": [
#                     {
#                         "match": {
#                             "paragraph_text": {
#                                 "query": question,
#                                 "boost": sum(idf_scores)
#                             }
#                         }
#                     },
#                     {
#                         "multi_match": {
#                             "query": question,
#                             "type": "most_fields",
#                             "fields": ["paragraph_text"],
#                             "operator": "and",
#                             "boost": sum(idf_scores)
#                         }
#                     },
#                    {
#                        "multi_match": {
#                            "query"                : question,
#                            "type"                 : "most_fields",
#                            "fields"               : ["paragraph_text"],
#                            "minimum_should_match" : "50%"
#                        }
#                    }
#                 ]+the_shoulds,
#                 "minimum_should_match": 1,
#             }
#         }
#     }
#     res         = es.search(index=doc_index, body=bod, request_timeout=120)
#     return res['hits']['hits']

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

def get_all_quests(dataset):
    ################################################
    questions_index = 'natural_questions_q_0_1'
    questions_map   = "natural_questions_q_map_0_1"
    es              = Elasticsearch(['{}:9200'.format(elk_ip)], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    if(dataset is None):
        bod = {}
    else:
        bod             = {"query": {"bool": {"must": [{"term": {"dataset": dataset}}]}}}
    items           = scan(es, query=bod, index=questions_index, doc_type=questions_map)
    total           = es.count(index=questions_index, body=bod)['count']
    ################################################
    return items, total

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def load_idfs_from_df(df_path):
    print('Loading IDF tables')
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    N   = 2684631
    idf = dict(
        [
            (
                item[0],
                math.log((N*1.0) / (1.0*item[1]))
            )
            for item in df.items()
        ]
    )
    ##############
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    ##############
    print('Loaded idf tables with max idf {}'.format(max_idf))
    return idf, max_idf

# idf, max_idf    = load_idfs_from_df('/home/dpappas/NQ_data/NQ_my_tokenize_df.pkl')

training7b_train_json = {'questions': []}
bm25_top100_train_pkl = {'queries': []}
bm25_docset_train_pkl = {}

#############################################################################

# all_dev_quests,     total_dev_quests    = get_all_quests('dev')
# all_train_quests,   total_train_quests  = get_all_quests('train')
all_quests,         total_quests        = get_all_quests(None)

#############################################################################

# print(total_dev_quests)
# print(total_train_quests)
print(total_quests)

#############################################################################

pbar                                    = tqdm(all_quests, total=total_quests)
my_counter, zero_count                  = 0, 0

#############################################################################

for quest in pbar:
    quest           = quest['_source']
    qtext           = quest['question']
    short_answer    = quest['short_answer']
    long_answer     = BeautifulSoup(quest['long_answer'], 'lxml').text.strip()
    ####################
    # pprint(quest)
    # exit()
    if ('<table>' in quest['long_answer'].lower()):
        continue
    else:
        # DONE
        bm25_100_datum  = {
            'num_rel'               : 0,
            'num_rel_ret'           : 0,
            'num_ret'               : 100,
            'query_id'              : str(quest["example_id"]),
            'query_text'            : quest['question'],
            'relevant_documents'    : [],
            'retrieved_documents'   : []
        }
        #######################
        q_data          = {
            "id"            : str(quest["example_id"]),
            "body"          : quest['question'],
            "documents"     : [],
            "snippets"      : []
        }
        #######################
        qtoks           = tokenize(qtext)
        # idf_scores      = [idf_val(w, idf, max_idf) for w in qtoks]
        # all_retr_docs   = get_first_n_11(qtoks, 100, idf_scores)
        all_retr_docs   = get_first_n(qtext, 100)
        ##########################################
        kept_docs   = {}
        rank        = 0
        for ret_doc in all_retr_docs:
            rank += 1
            # pprint(ret_doc)
            ##################################################################
            kept_docs[ret_doc['_id']]   = {
                u'pmid'             : ret_doc['_id'],
                u'abstractText'     : ret_doc['_source']['paragraph_text'],
                u'title'            : ret_doc['_source']['document_title']
            }
            ##################################################################
            paragraph_text              = ' '.join(tokenize(ret_doc['_source']['paragraph_text']))
            ############################################
            is_relevant = False
            if(short_answer in ret_doc['_source']['paragraph_text']):
                similarity = similar(paragraph_text, long_answer)
                if(similarity > 0.8 ):
                    # DOC IS RELEVANT
                    is_relevant                     = True
                    bm25_100_datum['num_rel']       += 1
                    bm25_100_datum['num_rel_ret']   += 1
                    bm25_100_datum['relevant_documents'].append(ret_doc['_id'])
                    q_data['documents'].append("http://www.ncbi.nlm.nih.gov/pubmed/{}".format(ret_doc['_id']))
                    q_data['snippets'].append(
                        {
                            "offsetInBeginSection"      : ret_doc['_source']['paragraph_text'].index(short_answer), # 131,
                            "offsetInEndSection"        : ret_doc['_source']['paragraph_text'].index(short_answer)+len(short_answer), # 358,
                            "text"                      : short_answer,
                            "beginSection"              : "abstract",
                            "document"                  : "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(ret_doc['_id']),
                            "endSection"                : "abstract"
                        }
                    )
                else:
                    # DOC IS NOT RELEVANT
                    pass
            else:
                # DOC IS NOT RELEVANT
                pass
            ############################################
            bm25_100_datum['retrieved_documents'].append({
                    u'bm25_score'       : ret_doc['_score'],
                    u'doc_id'           : ret_doc['_id'],
                    u'is_relevant'      : is_relevant,
                    u'norm_bm25_score'  : -1.0,
                    u'rank'             : rank
                })
            ############################################
        ##########################################
        if(bm25_100_datum['num_rel_ret']==0):
            zero_count += 1
        else:
            # KEEP IT IN THE DATASET
            # update docs
            bm25_docset_train_pkl.update(kept_docs)
            # update bm25
            bm25_top100_train_pkl['queries'].append(bm25_100_datum)
            # update questions
            training7b_train_json['questions'].append(q_data)
    # if(len(bm25_top100_train_pkl['queries'])==2):
    #     break

#############################################################################

# print(20 * '=')
# pprint(bm25_docset_train_pkl)  # PERFECT
# print(20 * '=')
# pprint(bm25_top100_train_pkl)  # PERFECT
# print(20 * '=')
# pprint(training7b_train_json)
# print(20 * '=')

#############################################################################

bioasq7_data    = training7b_train_json
train_data      = bm25_top100_train_pkl
train_docs      = bm25_docset_train_pkl

########################################################
# SPLIT: train | dev | test : 0.8 | 0.1 | 0.1
########################################################

dev_from    = int(len(train_data['queries']) * 0.8)
dev_to      = int(len(train_data['queries']) * 0.9)

dev_data    = {'queries': train_data['queries'][dev_from:dev_to]}
test_data   = {'queries': train_data['queries'][dev_to:]}
train_data  = {'queries': train_data['queries'][:dev_from]}

########################################################

odir = '/home/dpappas/NQ_data/'

with open(os.path.join(odir, 'NQ_training7b.train.dev.test.json'), 'w') as f:
    f.write(json.dumps(training7b_train_json, indent=4, sort_keys=True))

pickle.dump(bm25_docset_train_pkl,  open(os.path.join(odir, 'NQ_bioasq7_bm25_docset_top100.train.dev.test.pkl'),  'wb'))

pickle.dump(train_data,             open(os.path.join(odir, 'NQ_bioasq7_bm25_top100.train.pkl'), 'wb'))
pickle.dump(dev_data,               open(os.path.join(odir, 'NQ_bioasq7_bm25_top100.dev.pkl'),   'wb'))
pickle.dump(test_data,              open(os.path.join(odir, 'NQ_bioasq7_bm25_top100.test.pkl'),  'wb'))

#############################################################################

# FORMATS

'''
## training7b.train.json
# {
#     "questions": [
#           {
#                 "id": "55031181e9bde69634000014",
#                 "body": "Is Hirschsprung disease a mendelian or a multifactorial disorder?",
#                 "documents": [
#                       "http://www.ncbi.nlm.nih.gov/pubmed/15829955",
#                       ...
#                 ],
#                 "snippets" : [
#                     {
#                         "offsetInBeginSection": 131,
#                         "offsetInEndSection": 358,
#                         "text": "Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes",
#                         "beginSection": "abstract",
#                         "document": "http://www.ncbi.nlm.nih.gov/pubmed/15829955",
#                         "endSection": "abstract"
#                     }, ...
#                 ]
#           }
'''

'''
## bioasq7_bm25_top100.train.pkl
# {
#     'queries':[
#         {
#             'num_rel': 9,
#             'num_rel_ret': 6,
#             'num_ret': 100,
#             'query_id': u'55031181e9bde69634000014',
#             'query_text': u'Is Hirschsprung disease a mendelian or a multifactorial disorder?',
#             'relevant_documents': [ '12239580', ...],
#             'retrieved_documents': [
#                 {
#                     u'bm25_score': 7.02374051,
#                     u'doc_id': u'15617541',
#                     u'is_relevant': True,
#                     u'norm_bm25_score': 3.7238850702205664,
#                     u'rank': 1
#                 }, ...
#             ]
#         }, ...
#     ]
# }
'''

'''
## bioasq7_bm25_docset_top100.train.pkl
# {
#     u'20176987' : {
#         {
#             u'abstractText': u'BACKGROUND\nAging and aging-related disorders impair the survival and differentiation potential of bone marrow mesenchymal stem cells (MSCs) and limit their therapeutic efficacy. Induced pluripotent stem cells (iPSCs) may provide an alternative source of functional MSCs for tissue repair. This study aimed to generate and characterize human iPSC-derived MSCs and to investigate their biological function for the treatment of limb ischemia.\n\n\nMETHODS AND RESULTS\nHuman iPSCs were induced to MSC differentiation with a clinically compliant protocol. Three monoclonal, karyotypically stable, and functional MSC-like cultures were successfully isolated using a combination of CD24(-) and CD105(+) sorting. They did not express pluripotent-associated markers but displayed MSC surface antigens and differentiated into adipocytes, osteocytes, and chondrocytes. Transplanting iPSC-MSCs into mice significantly attenuated severe hind-limb ischemia and promoted vascular and muscle regeneration. The benefits of iPSC-MSCs on limb ischemia were superior to those of adult bone marrow MSCs. The greater potential of iPSC-MSCs may be attributable to their superior survival and engraftment after transplantation to induce vascular and muscle regeneration via direct de novo differentiation and paracrine mechanisms.\n\n\nCONCLUSIONS\nFunctional MSCs can be clonally generated, beginning at a single-cell level, from human iPSCs. Patient-specific iPSC-MSCs can be prepared as an "off-the-shelf" format for the treatment of tissue ischemia.',
#             u'pmid': u'20176987',
#             u'publicationDate': u'2010-03-09',
#             u'title': u'Functional mesenchymal stem cells derived from human induced pluripotent stem cells attenuate limb ischemia in mice.'
#         }
#     },
#     ...
# }
'''

'''

#################
train_data  = []
dev_data    = []
#################
for quest in pbar:
    qtext           = quest['_source']['question']
    short_answer    = quest['_source']['short_answer']
    long_answer     = BeautifulSoup(quest['_source']['long_answer'], 'lxml').text.strip()
    ####################
    if ('<table>' in quest['_source']['long_answer'].lower()):
        continue
    ####################
    all_retr_docs   = get_first_n(qtext, 100)
    ####################
    relevant_docs, irrelevant_docs = [], []
    for ret_doc in all_retr_docs:
        paragraph_text  = ' '.join(tokenize(ret_doc['_source']['paragraph_text']))
        ############################################
        if(short_answer in ret_doc['_source']['paragraph_text']):
            similarity = similar(paragraph_text, long_answer)
            if(similarity > 0.8 ):
                relevant_docs.append(ret_doc)
            else:
                irrelevant_docs.append(ret_doc)
        else:
            irrelevant_docs.append(ret_doc)
    if(len(relevant_docs)==0):
        zero_count += 1
    ####################
    quest['_source']['relevant_docs']   = relevant_docs
    quest['_source']['irrelevant_docs'] = irrelevant_docs
    ####################
    if(quest['_source']['dataset'] == 'train'):
        train_data.append(quest)
    else:
        dev_data.append(quest)
    ####################
    pbar.set_description('{} - {}'.format(zero_count, total_train_quests))

#############################################################################

pickle.dump(train_data, open('/home/dpappas/NQ_data/NQ_train_data.pkl', 'wb'), protocol=2)
pickle.dump(dev_data,   open('/home/dpappas/NQ_data/NQ_dev_data.pkl',   'wb'), protocol=2)
'''








