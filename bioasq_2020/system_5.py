
__author__ = 'Dimitris'

import  numpy as np
from    tqdm  import tqdm
import  pickle, os, json
from adhoc_vectorizer import get_sentence_vecs
# from my_sentence_splitting import get_sents
from nltk.tokenize import sent_tokenize
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity

in_dir      = '/home/dpappas/bioasq_all/bioasq8/data/test_batch_1/bioasq8_bm25_top100/'

docs_data   = pickle.load(open(os.path.join(in_dir, 'bioasq8_bm25_docset_top100.test.pkl'), 'rb'))
ret_data    = pickle.load(open(os.path.join(in_dir, 'bioasq8_bm25_top100.test.pkl'), 'rb'))

# pprint(ret_data['queries'][0])

system_subm_data    = {'questions':[]}

# {
#             "body": "Describe the mechanism of action of a drug Elagolix.",
#             "documents": [
#                 "http://www.ncbi.nlm.nih.gov/pubmed/30551159",
#                 "http://www.ncbi.nlm.nih.gov/pubmed/27021205",
#                 "http://www.ncbi.nlm.nih.gov/pubmed/28737050",
#                 "http://www.ncbi.nlm.nih.gov/pubmed/19033369",
#                 "http://www.ncbi.nlm.nih.gov/pubmed/28525302",
#                 "http://www.ncbi.nlm.nih.gov/pubmed/30320043",
#                 "http://www.ncbi.nlm.nih.gov/pubmed/28323948",
#                 "http://www.ncbi.nlm.nih.gov/pubmed/28579415",
#                 "http://www.ncbi.nlm.nih.gov/pubmed/30303923",
#                 "http://www.ncbi.nlm.nih.gov/pubmed/28255765"
#             ],
#             "id": "5e2900368b3851296d000001",
#             "snippets": [
#                 {
#                     "beginSection": "abstract",
#                     "document": "http://www.ncbi.nlm.nih.gov/pubmed/30551159",
#                     "endSection": "abstract",
#                     "offsetInBeginSection": 0,
#                     "offsetInEndSection": 187,
#                     "text": "Elagolix, an orally active non-peptidic GnRH antagonist, has been approved by the Food and Drug Administration for the management of moderate to severe pain associated with endometriosis."
#                 },}

for quer in tqdm(ret_data['queries']):
    qid     = quer['query_id']
    qtext   = quer['query_text']
    qvecs   = get_sentence_vecs(qtext)
    if(qvecs is None):
        continue
    #############################################
    subm_data = {
        "body"      : qtext,
        "documents" : [],
        "id"        : qtext,
        "snippets"  : []
    }
    #############################################
    sent_res    = []
    #############################################
    for ret_doc in quer['retrieved_documents']:
        norm_bm25   = ret_doc['norm_bm25_score']
        doc_id      = ret_doc['doc_id']
        #############################################
        abstract    = ' '.join([
            token for token in docs_data[doc_id]['abstractText'].split()
            if(not token.startswith('__') and not token.endswith('__'))
        ])
        # abs_sents   = get_sents(abstract)
        abs_sents   = sent_tokenize(abstract)
        title       = ' '.join([
            token for token in docs_data[doc_id]['title'].split()
            if(not token.startswith('__') and not token.endswith('__'))
        ])
        # tit_sents   = get_sents(title)
        tit_sents   = sent_tokenize(title)
        #############################################
        for sent in tit_sents:
            svecs       = get_sentence_vecs(sent)
            if (svecs is None):
                continue
            sim         = cosine_similarity(qvecs, svecs).max()
            offset_from = title.index(sent)
            offset_to   = offset_from + len(sent)
            sent_res.append(
                (sim, doc_id, norm_bm25, 'title', sent, offset_from, offset_to)
            )
        for sent in abs_sents:
            svecs       = get_sentence_vecs(sent)
            if (svecs is None):
                continue
            sim         = cosine_similarity(qvecs, svecs).max()
            offset_from = abstract.index(sent)
            offset_to   = offset_from + len(sent)
            sent_res.append(
                (sim, doc_id, norm_bm25, 'abstract', sent, offset_from, offset_to)
            )
    #############################################
    sent_res    = sorted(sent_res, key=lambda x: x[0] * x[2], reverse=True)
    [sent_res]
    pprint(sent_res[:10])
    #############################################
    break




