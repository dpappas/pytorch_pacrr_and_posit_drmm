
import os, sys, json, pickle, random
from pprint import pprint

def load_all_data(dataloc):
    print('loading pickle data')
    #
    with open(dataloc + 'trainining7b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict((q['id'], q) for q in bioasq6_data['questions'])
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
    return dev_data, dev_docs, train_data, train_docs, bioasq6_data

odir = '/home/dpappas/fast_bert_models/doc_rerank/'
if(not os.path.exists(odir)):
    os.makedirs(odir)

with open(os.path.join(odir, 'labels.csv'), 'w') as f:
    f.write('pos')
    f.write('\n')
    f.write('neg')
    f.close()

dataloc = '/home/dpappas/bioasq_all/bioasq7_data/'
(dev_data, dev_docs, train_data, train_docs, bioasq6_data) = load_all_data(dataloc=dataloc)

# TRAIN DATA

index = 0
lines = []
for query in train_data['queries']:
    qtext       = query['query_text'].replace(',', '').replace('\n', ' ')
    rel_docs    = query['relevant_documents']
    retr_docs   = [d['doc_id'] for d in query['retrieved_documents']]
    for doc_id in retr_docs:
        label       = 'pos' if(doc_id in rel_docs) else 'neg'
        doc_text    = train_docs[doc_id]['title'].replace('\n', ' ') + ' <title> ' + train_docs[doc_id]['abstractText'].replace('\n', ' ')
        doc_text    = doc_text.replace(',', '')
        line        = [str(index), qtext + ' ### ' + doc_text, label]
        lines.append(line)
    index += 1

random.shuffle(lines)

with open(os.path.join(odir, 'train.csv'), 'w') as f:
    f.write(','.join(['index', 'text', 'label']))
    f.write('\n')
    for line in lines:
        f.write(','.join(line))
        f.write('\n')
    f.close()

# VAL DATA

index = 0
lines = []
for query in dev_data['queries']:
    qtext       = query['query_text'].replace(',', '').replace('\n', ' ')
    rel_docs    = query['relevant_documents']
    retr_docs   = [d['doc_id'] for d in query['retrieved_documents']]
    for doc_id in retr_docs:
        label       = 'pos' if(doc_id in rel_docs) else 'neg'
        doc_text    = dev_docs[doc_id]['title'].replace('\n', ' ') + ' <title> ' + dev_docs[doc_id]['abstractText'].replace('\n', ' ')
        doc_text    = doc_text.replace(',', '')
        line        = [str(index), qtext + ' ### ' + doc_text, label]
        lines.append(line)
    index += 1

random.shuffle(lines)

with open(os.path.join(odir, 'val.csv'), 'w') as f:
    f.write(','.join(['index', 'text', 'label']))
    f.write('\n')
    for line in lines:
        f.write(','.join(line))
        f.write('\n')
    f.close()

