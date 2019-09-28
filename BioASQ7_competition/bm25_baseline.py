
import pickle, json, sys
from pprint import pprint
from    nltk.tokenize               import sent_tokenize

w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
dataloc             = '/home/dpappas/bioasq_all/bioasq7_data/'

b       = 1 #sys.argv[1]
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

print('loading idfs')
with open(idf_pickle_path, 'rb') as f:
    idf = pickle.load(f)

###########################################################

bm25_data   = {'questions': []}
for q in test_data['queries']:
    ###############################
    q_text      = q['query_text']
    docs        = [d['doc_id'] for d in q['retrieved_documents']][:10]
    all_sents   = []
    for did in docs:
        all_sents.extend(sent_tokenize(test_docs[did]['title']) + sent_tokenize(test_docs[did]['abstractText']))
    ###############################
    bm25_data['questions'].append(
        {
            "body"      : "n/a",
            "id"        : q['query_id'],
            "documents" : ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(d['doc_id']) for d in docs]
        }
    )

with open(f_out, 'w') as f:
    f.write(json.dumps(bm25_data, indent=4, sort_keys=False))
    f.close()

'''

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/bm25.json \
| grep "^MAP documents:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/bm25.json \
| grep "^MAP documents:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/bm25.json \
| grep "^MAP documents:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_4/BioASQ-task7bPhaseB-testset4 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_4/bm25.json \
| grep "^MAP documents:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
/home/dpappas/bioasq_all/bioasq7/data/test_batch_5/BioASQ-task7bPhaseB-testset5 \
/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_5/bm25.json \
| grep "^MAP documents:"

'''


