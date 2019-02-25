
import pickle, json, os
from pprint import pprint
from nltk import sent_tokenize
from tqdm import tqdm

def RemoveTrainLargeYears(data, doc_text):
  for i in range(len(data['queries'])):
    hyear = 1900
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      if data['queries'][i]['retrieved_documents'][j]['is_relevant']:
        doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
        year = doc_text[doc_id]['publicationDate'].split('-')[0]
        if year[:1] == '1' or year[:1] == '2':
          if int(year) > hyear:
            hyear = int(year)
    j = 0
    while True:
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      year = doc_text[doc_id]['publicationDate'].split('-')[0]
      if (year[:1] == '1' or year[:1] == '2') and int(year) > hyear:
        del data['queries'][i]['retrieved_documents'][j]
      else:
        j += 1
      if j == len(data['queries'][i]['retrieved_documents']):
        break
  return data

def RemoveBadYears(data, doc_text, train):
  for i in range(len(data['queries'])):
    j = 0
    while True:
      doc_id    = data['queries'][i]['retrieved_documents'][j]['doc_id']
      year      = doc_text[doc_id]['publicationDate'].split('-')[0]
      ##########################
      # Skip 2017/2018 docs always. Skip 2016 docs for training.
      # Need to change for final model - 2017 should be a train year only.
      # Use only for testing.
      if year == '2017' or year == '2018' or (train and year == '2016'):
      #if year == '2018' or (train and year == '2017'):
        del data['queries'][i]['retrieved_documents'][j]
      else:
        j += 1
      ##########################
      if j == len(data['queries'][i]['retrieved_documents']):
        break
  return data

def load_all_data(dataloc):
    print('loading pickle data')
    #
    with open(dataloc+'BioASQ-trainingDataset6b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict( (q['id'], q) for q in bioasq6_data['questions'] )
    #
    with open(dataloc + 'bioasq_bm25_top100.test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.test.pkl', 'rb') as f:
        test_docs = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.dev.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.dev.pkl', 'rb') as f:
        dev_docs = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.train.pkl', 'rb') as f:
        train_docs = pickle.load(f)
    print('loading words')
    #
    train_data  = RemoveBadYears(train_data, train_docs, True)
    train_data  = RemoveTrainLargeYears(train_data, train_docs)
    dev_data    = RemoveBadYears(dev_data, dev_docs, False)
    test_data   = RemoveBadYears(test_data, test_docs, False)
    #
    return test_data, test_docs, dev_data, dev_docs, train_data, train_docs, bioasq6_data

# w2v_bin_path        = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/dpappas/for_ryan/fordp/idf.pkl'
dataloc             = '/home/dpappas/for_ryan/'
out_dataloc         = '/home/dpappas/for_ryan_clean/'

if not os.path.exists(out_dataloc):
    os.makedirs(out_dataloc)

(test_data, test_docs, dev_data, dev_docs, train_data, train_docs, bioasq6_data) = load_all_data(dataloc=dataloc)

# print(test_data.keys())
bioasq6_data_2  = {}
deleted_pmids   = []
deleted_qids    = []
max_sent_len    = 1
for qid in tqdm(bioasq6_data):
    if('snippets' not in bioasq6_data[qid]):
        deleted_qids.append(qid)
        continue
    del_pmids                       = [snip['document'] for snip in bioasq6_data[qid]['snippets'] if (len(sent_tokenize(snip['text']))>max_sent_len)]
    held_snips                      = [snip for snip in bioasq6_data[qid]['snippets'] if(len(sent_tokenize(snip['text']))<=max_sent_len)]
    ret_pmids                       = [snip['document'] for snip in held_snips]
    deleted_pmids.extend(del_pmids)
    bioasq6_data[qid]['documents']  = [d for d in bioasq6_data[qid]['documents'] if(d in ret_pmids)]
    if(len(bioasq6_data[qid]['documents']) != 0):
        bioasq6_data_2[qid] = bioasq6_data[qid]
    else:
        deleted_qids.append(qid)

############

test_data_2 = {'queries':[]}
for datum in tqdm(test_data['queries']):
    if(datum['query_id'] in deleted_qids):
        continue
    datum['relevant_documents'] = [rd for rd in datum['relevant_documents'] if(rd not in deleted_pmids)]
    if(len(datum['relevant_documents'])==0):
        continue
    else:
        datum['retrieved_documents'] = [
            rd for rd in datum['retrieved_documents'] if (rd['doc_id'] not in deleted_pmids)
        ]
        test_data_2['queries'].append(datum)

############

dev_data_2 = {'queries':[]}
for datum in tqdm(dev_data['queries']):
    if(datum['query_id'] in deleted_qids):
        continue
    datum['relevant_documents'] = [rd for rd in datum['relevant_documents'] if(rd not in deleted_pmids)]
    if(len(datum['relevant_documents'])==0):
        continue
    else:
        datum['retrieved_documents'] = [
            rd for rd in datum['retrieved_documents'] if (rd['doc_id'] not in deleted_pmids)
        ]
        dev_data_2['queries'].append(datum)

############

train_data_2 = {'queries':[]}
for datum in tqdm(train_data['queries']):
    if(datum['query_id'] in deleted_qids):
        continue
    datum['relevant_documents'] = [rd for rd in datum['relevant_documents'] if(rd not in deleted_pmids)]
    if(len(datum['relevant_documents'])==0):
        continue
    else:
        datum['retrieved_documents'] = [
            rd for rd in datum['retrieved_documents'] if (rd['doc_id'] not in deleted_pmids)
        ]
        train_data_2['queries'].append(datum)

############

train_docs_2    = dict([item for item in train_docs.items() if(item[0] not in deleted_pmids)])
dev_docs_2      = dict([item for item in dev_docs.items()   if(item[0] not in deleted_pmids)])
test_docs_2     = dict([item for item in test_docs.items()  if(item[0] not in deleted_pmids)])

print(len(bioasq6_data), len(bioasq6_data_2))
print(len(list(set(deleted_pmids))))
print(len(list(set(deleted_qids))))
print(len(test_data['queries']), len(test_data_2['queries']))
print(len(dev_data['queries']), len(dev_data_2['queries']))
print(len(train_data['queries']), len(train_data_2['queries']))
print(len(train_docs), len(train_docs_2))
print(len(dev_docs), len(dev_docs_2))
print(len(test_docs), len(test_docs_2))

pickle.dump(dev_data_2, open(os.path.join(out_dataloc, 'bioasq_bm25_top100.dev.pkl'), 'wb'), protocol=2)
pickle.dump(test_data_2, open(os.path.join(out_dataloc, 'bioasq_bm25_top100.test.pkl'), 'wb'), protocol=2)
pickle.dump(train_data_2, open(os.path.join(out_dataloc, 'bioasq_bm25_top100.train.pkl'), 'wb'), protocol=2)
#
pickle.dump(dev_docs_2, open(os.path.join(out_dataloc, 'bioasq_bm25_docset_top100.dev.pkl'), 'wb'), protocol=2)
pickle.dump(test_docs_2, open(os.path.join(out_dataloc, 'bioasq_bm25_docset_top100.test.pkl'), 'wb'), protocol=2)
pickle.dump(train_docs_2, open(os.path.join(out_dataloc, 'bioasq_bm25_docset_top100.train.pkl'), 'wb'), protocol=2)
#
with open(os.path.join(out_dataloc, 'BioASQ-trainingDataset6b.json'), 'w') as f:
    f.write(json.dumps(bioasq6_data_2, indent=4, sort_keys=True))
    f.close()
