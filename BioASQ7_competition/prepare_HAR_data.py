#!/usr/bin/env python
# -*- coding: utf-8 -*-

import  json
import  numpy                       as np
import  pickle
from    pprint                      import pprint
import  re
import  nltk
from nltk.tokenize import sent_tokenize

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
softmax     = lambda z: np.exp(z) / np.sum(np.exp(z))
stopwords   = nltk.corpus.stopwords.words("english")

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

def snip_is_relevant(one_sent, gold_snips):
    # print one_sent
    # pprint(gold_snips)
    return int(
        any(
            [
                (one_sent.encode('ascii', 'ignore')  in gold_snip.encode('ascii','ignore'))
                or
                (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii','ignore'))
                for gold_snip in gold_snips
            ]
        )
    )

def get_snips(quest_id, gid, bioasq6_data):
    good_snips = []
    if('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            if(sn['document'].endswith(gid)):
                good_snips.extend(sent_tokenize(sn['text']))
    return good_snips

dataloc = '/home/dpappas/bioasq_all/bioasq7_data/'

(dev_data, dev_docs, train_data, train_docs, bioasq7_data) = load_all_data(dataloc)

pprint(dev_data.keys())

for item in dev_data['queries']:
    query_text  = item['query_text']
    quest_id    = item['query_id']
    for retr in item['retrieved_documents']:
        good_snips  = get_snips(quest_id, retr['doc_id'], bioasq7_data)
        good_snips  = [' '.join(bioclean(sn)) for sn in good_snips]
        # print(good_snips)
        doc         = dev_docs[retr['doc_id']]
        title       = doc['title']
        abs         = doc['abstractText']
        all_sents   = sent_tokenize(title) + sent_tokenize(abs)
        for one_sent in all_sents:
            tag = snip_is_relevant(' '.join(bioclean(one_sent)), good_snips)
            print('\t'.join([str(tag), query_text, one_sent]))


pprint(dev_data['queries'][0])

