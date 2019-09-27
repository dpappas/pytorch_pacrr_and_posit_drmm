
import pickle, json

w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
dataloc             = '/home/dpappas/bioasq_all/bioasq7_data/'

def load_all_data(dataloc):
    print('loading pickle data')
    #
    with open(dataloc+'trainining7b.json', 'r') as f:
        bioasq7_data = json.load(f)
        bioasq7_data = dict((q['id'], q) for q in bioasq7_data['questions'])
    #
    with open(dataloc + 'bioasq7_bm25_top100.dev.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_docset_top100.dev.pkl', 'rb') as f:
        dev_docs = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_top100.train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_docset_top100.train.pkl', 'rb') as f:
        train_docs = pickle.load(f)
    print('loading words')
    return dev_data, dev_docs, train_data, train_docs, bioasq7_data


(dev_data, dev_docs, train_data, train_docs, bioasq7_data) = load_all_data(dataloc)




