
from rank_bm25 import BM25Okapi
import pickle, json, sys, re
import numpy as np
from pprint import pprint
from    nltk.tokenize               import sent_tokenize

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
dataloc             = '/home/dpappas/bioasq_all/bioasq7_data/'

b       = sys.argv[1]
f_in1   = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseA-testset{}'.format(b, b)
f_in2   = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'.format(b)
f_in3   = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl'.format(b)
f_out   = '/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_{}/bm25.json'.format(b)

###########################################################
print('loading pickle data')
with open(f_in1, 'r') as f:
    bioasq7_data = json.load(f)
    for q in bioasq7_data['questions']:
        if("documents" not in q):
            q["documents"]  = []
        if("snippets" not in q):
            q["snippets"]   = []
    bioasq7_data = dict((q['id'], q) for q in bioasq7_data['questions'])

with open(f_in2, 'rb') as f:
    test_data = pickle.load(f)

with open(f_in3, 'rb') as f:
    test_docs = pickle.load(f)

# print('loading idfs')
# with open(idf_pickle_path, 'rb') as f:
#     idf = pickle.load(f)

###########################################################

bm25_data   = {'questions': []}
for q in test_data['queries']:
    ###############################
    q_text      = q['query_text']
    docs        = [d['doc_id'] for d in q['retrieved_documents']]
    all_sents       = []
    all_text_sents  = []
    for did in docs:
        for sent in sent_tokenize(test_docs[did]['title']):
            all_sents.append(
                (
                    "title",
                    "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(did),
                    "title",
                    test_docs[did]['title'].index(sent),
                    test_docs[did]['title'].index(sent)+len(sent),
                    sent
                )
            )
            all_text_sents.append(sent)
        for sent in sent_tokenize(test_docs[did]['abstractText']):
            all_sents.append(
                (
                    "abstract",
                    "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(did),
                    "abstract",
                    test_docs[did]['abstractText'].index(sent),
                    test_docs[did]['abstractText'].index(sent)+len(sent),
                    sent
                )
            )
            all_text_sents.append(sent)
    ###############################
    tokenized_corpus    = [bioclean(doc) for doc in all_text_sents]
    bm25                = BM25Okapi(tokenized_corpus)
    doc_scores          = bm25.get_scores(bioclean(q_text))
    max_inds            = (np.array(doc_scores)).argsort()[-10:][::-1].tolist()
    retr_sents          = [all_sents[index] for index in max_inds]
    ###############################
    retr_snips          = [
        {
            "beginSection"          : snipi[0],
            "document"              : snipi[1],
            "endSection"            : snipi[2],
            "offsetInBeginSection"  : snipi[3],
            "offsetInEndSection"    : snipi[4],
            "text"                  : snipi[5],
        }
        for snipi in retr_sents
    ]
    bm25_data['questions'].append(
        {
            "body"      : "n/a",
            "id"        : q['query_id'],
            "documents" : ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(doc_id) for doc_id in docs[:10]],
            "snippets"  : retr_snips
        }
    )

with open(f_out, 'w') as f:
    f.write(json.dumps(bm25_data, indent=4, sort_keys=False))
    f.close()

'''

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/bm25.json \
| grep "^MAP "

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/bm25.json \
| grep "^MAP "

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/bm25.json \
| grep "^MAP "

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_4/BioASQ-task7bPhaseB-testset4 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_4/bm25.json \
| grep "^MAP "

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_5/BioASQ-task7bPhaseB-testset5 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_5/bm25.json \
| grep "^MAP "

'''


