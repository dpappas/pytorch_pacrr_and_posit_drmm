
import pickle, json
import numpy as np
import re
from nltk.tokenize import sent_tokenize

# from fuzzywuzzy import fuzz
# # # fuzz.ratio("this is a test", "this is a test!")


bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())

def tf(term, document):
    tf = 0
    for word in document:
        if word == term:
            tf += 1
    if len(document) == 0:
        return tf
    else:
        return tf / len(document)

def similarity_score(query, document, k1, b, idf_scores, avgdl, normalize, mean, deviation, rare_word):
    score = 0
    for query_term in query:
        if query_term not in idf_scores:
            score += rare_word * (
                    (tf(query_term, document) * (k1 + 1)) /
                    (
                            tf(query_term, document) +
                            k1 * (1 - b + b * (len(document) / avgdl))
                    )
            )
        else:
            score += idf_scores[query_term] * ((tf(query_term, document) * (k1 + 1)) / (
                        tf(query_term, document) + k1 * (1 - b + b * (len(document) / avgdl))))
    if normalize:
        return ((score - mean) / deviation)
    else:
        return score

def load_idfs(idf_path):
    print('Loading IDF tables')
    ###############################
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    print('Loaded idf tables with max idf {}'.format(max_idf))
    ###############################
    return idf, max_idf

k1, b = 1.2, 0.75
avgdl=20.47079583909152
mean=0.6158675062087192
deviation=1.205199607538813

idf_pickle_path         = '/home/dpappas/bioasq_all/idf.pkl'
idf, max_idf            = load_idfs(idf_pickle_path)

# BM25score = similarity_score(q_toks, bioclean(sent).split(), k1, b, idf, avgdl, False, 0, 0, max_idf)

def snip_is_relevant(one_sent, gold_snips):
    # return int(
    #     any(
    #         [
    #             fuzz.ratio(one_sent, gold_snip) > 0.95 for gold_snip in gold_snips
    #         ]
    #     )
    # )
    return int(
        any(
            [
                # (one_sent.encode('ascii', 'ignore')  in gold_snip.encode('ascii','ignore'))
                # or
                # (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii','ignore'))
                (one_sent in gold_snip or gold_snip in one_sent) and
                (
                        len(one_sent) < len(gold_snip)+6 and
                        len(one_sent) > len(gold_snip) - 6
                )
                for gold_snip in gold_snips
            ]
        )
    )

for batch_no in range(1,6):
    le_docs = pickle.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl'.format(batch_no),'rb'))
    d   = pickle.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'.format(batch_no), 'rb'))
    d2  = json.load(open('bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(batch_no,batch_no)))
    ##################################################################################################
    id2rel      = {}
    id2relsnip  = {}
    for q in d2['questions']:
        id2rel[q['id']]     = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in q['documents']]
        id2relsnip[q['id']] = [bioclean(snip['text'].strip()) for snip in q['snippets']]
    ##################################################################################################
    recalls         = []
    recalls_snip    = []
    # for quer in d['queries']:
    #     retr_ids     = [doc['doc_id'] for doc in quer['retrieved_documents'][:10] if doc['doc_id'] in id2rel[quer['query_id']]]
    #     found         = float(len(retr_ids))
    #     tot         = float(len(id2rel[quer['query_id']]))
    #     recalls.append(found/tot)
    for quer in d['queries']:
        retr_doc_ids    = [doc['doc_id'] for doc in quer['retrieved_documents'][:10]]
        retr_ids        = [doc_id for doc_id in retr_doc_ids if doc_id in id2rel[quer['query_id']]]
        found           = float(len(retr_ids))
        tot             = float(len(id2rel[quer['query_id']]))
        recalls.append(found/tot)
        #################
        all_sents       = []
        for doc_id in retr_doc_ids:
            all_sents.extend(sent_tokenize(le_docs[doc_id]['title']))
            all_sents.extend(sent_tokenize(le_docs[doc_id]['abstractText']))
        retr_snips          = list(set(all_sents))
        #################
        snipsWscore = [
            (
                snip,
                similarity_score(
                    bioclean(quer['query_text']),
                    bioclean(snip).split(), k1, b, idf, avgdl, False, 0, 0, max_idf
                )
            )
            for snip in retr_snips
        ]
        snipsWscore = sorted(snipsWscore, key=lambda x: x[1], reverse=True)
        retr_snips  = [t[0] for t in snipsWscore[:10]]
        #################
        if(len(id2relsnip[quer['query_id']]) != 0):
            retr_snips      = [t for t in retr_snips if snip_is_relevant(bioclean(t), id2relsnip[quer['query_id']])]
            recalls_snip.append(float(len(retr_snips))/float(len(id2relsnip[quer['query_id']])))
        #################
    ##################################################################################################
    print(np.average(recalls))
    print(np.average(recalls_snip))
    print(30 *'-')
    ##################################################################################################


for batch_no in range(1,6):
    le_docs = pickle.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl'.format(batch_no),'rb'))
    d   = pickle.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'.format(batch_no), 'rb'))
    d2  = json.load(open('bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(batch_no,batch_no)))
    ##################################################################################################
    id2rel      = {}
    id2relsnip  = {}
    for q in d2['questions']:
        id2rel[q['id']]     = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in q['documents']]
        id2relsnip[q['id']] = [bioclean(snip['text'].strip()) for snip in q['snippets']]
    ##################################################################################################
    recalls         = []
    recalls_snip    = []
    for quer in d['queries']:
        retr_doc_ids    = [doc['doc_id'] for doc in quer['retrieved_documents'][:100]]
        retr_ids        = [doc_id for doc_id in retr_doc_ids if doc_id in id2rel[quer['query_id']]]
        found           = float(len(retr_ids))
        if(found>10):
            found = 10.0
        tot             = float(len(id2rel[quer['query_id']]))
        recalls.append(found/tot)
        #################
        all_sents       = []
        for doc_id in retr_doc_ids:
            all_sents.extend(sent_tokenize(le_docs[doc_id]['title']))
            all_sents.extend(sent_tokenize(le_docs[doc_id]['abstractText']))
        retr_snips          = list(set(all_sents))
        #################
        if(len(id2relsnip[quer['query_id']]) != 0):
            retr_snips      = [t for t in retr_snips if snip_is_relevant(bioclean(t), id2relsnip[quer['query_id']])][:10]
            found           = float(len(retr_snips))
            if(found>10):
                found = 10.0
            tot             = float(len(id2relsnip[quer['query_id']]))
            recalls_snip.append(found/tot)
        #################
    ##################################################################################################
    print(np.average(recalls))
    print(np.average(recalls_snip))
    print(30 *'-')
    ##################################################################################################


for batch_no in range(1,6):
    le_out  = {"questions": []}
    le_docs = pickle.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl'.format(batch_no),'rb'))
    d       = pickle.load(open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'.format(batch_no), 'rb'))
    d2      = json.load(open('bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(batch_no,batch_no)))
    ##################################################################################################
    id2rel      = {}
    id2relsnip  = {}
    id2snips    = {}
    for q in d2['questions']:
        id2rel[q['id']]     = [doc.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for doc in q['documents']]
        id2relsnip[q['id']] = [bioclean(snip['text'].strip()) for snip in q['snippets']]
        id2snips[q['id']]   = q['snippets'][:10]
    ##################################################################################################
    recalls         = []
    recalls_snip    = []
    for quer in d['queries']:
        qo              = {
            "body"      : "n/a",
            "id"        : quer['query_id'],
            "documents" : [],
            "snippets"  : []
        }
        retr_doc_ids    = [doc['doc_id'] for doc in quer['retrieved_documents'][:100]]
        retr_ids        = [doc_id for doc_id in retr_doc_ids if doc_id in id2rel[quer['query_id']]][:10]
        qo['documents'] = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(did) for did in retr_ids]
        found           = float(len(retr_ids))
        tot             = float(len(id2rel[quer['query_id']]))
        recalls.append(found/tot)
        #################
        all_sents       = []
        for doc_id in retr_doc_ids:
            for sent in sent_tokenize(le_docs[doc_id]['title']):
                all_sents.append((sent, doc_id, 'title', le_docs[doc_id]['title'].index(sent)))
            for sent in sent_tokenize(le_docs[doc_id]['abstractText']):
                all_sents.append((sent, doc_id, 'abstractText', le_docs[doc_id]['abstractText'].index(sent)))
        retr_snips          = list(set(all_sents))
        #################
        qo['snippets'] = [
            t
            for t in id2snips[quer['query_id']]
            if t['document'] in qo['documents']
        ][:10]
        #################
        le_out['questions'].append(qo)
    ##################################################################################################
    print(np.average(recalls))
    print(np.average(recalls_snip))
    fout = '/home/dpappas/gold_{}'.format(batch_no)
    with open(fout, 'w') as f:
        gb = f.write(json.dumps(le_out, indent=4, sort_keys=True))
        f.close()
    print(30 *'-')
    ##################################################################################################

'''

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar \
evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1 \
/home/dpappas/gold_1 | grep ^MAP

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar \
evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2 \
/home/dpappas/gold_2 | grep ^MAP

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar \
evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3 \
/home/dpappas/gold_3 | grep ^MAP

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar \
evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_4/BioASQ-task7bPhaseB-testset4 \
/home/dpappas/gold_4 | grep ^MAP

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar \
evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_5/BioASQ-task7bPhaseB-testset5 \
/home/dpappas/gold_5 | grep ^MAP

'''


