#!/usr/bin/env python
# -*- coding: utf-8 -*-

# INSTRUCTIONS can be found in : https://github.com/mingzhu0527/HAR

import  os, re, json, random, pickle, nltk
import  numpy                       as np
from    tqdm                        import tqdm
from    pprint                      import pprint
from    nltk.tokenize import sent_tokenize

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

def do_one(data, docs, fname):
    lines = []
    for item in tqdm(data['queries']):
        query_text  = item['query_text']
        query_text  = query_text.replace('\t', ' ').replace('\n', ' ')
        query_text  = ' '.join(bioclean(query_text))
        if(len(query_text.strip())==0):
            continue
        quest_id    = item['query_id']
        for retr in item['retrieved_documents']:
            good_snips  = get_snips(quest_id, retr['doc_id'], bioasq7_data)
            good_snips  = [' '.join(bioclean(sn)) for sn in good_snips]
            doc         = docs[retr['doc_id']]
            title       = doc['title']
            abs         = doc['abstractText']
            all_sents   = sent_tokenize(title) + sent_tokenize(abs)
            for one_sent in all_sents:
                one_sent    = one_sent.replace('\t', ' ').replace('\n', ' ')
                one_sent    = ' '.join(bioclean(one_sent))
                if(len(one_sent.strip())==0):
                    continue
                tag         = snip_is_relevant(one_sent, good_snips)
                line        = '\t'.join([str(tag), query_text, one_sent])
                lines.append(line)
    random.shuffle(lines)
    with open(os.path.join(odir, fname), 'w') as fp:
        fp.write('\n'.join(lines))
        fp.close()

dataloc = '/home/dpappas/bioasq_all/bioasq7_data/'

(dev_data, dev_docs, train_data, train_docs, bioasq7_data) = load_all_data(dataloc)

odir = '/home/dpappas/HAR/data/bioasq7/'
if not os.path.exists(odir):
    os.makedirs(odir)

do_one(train_data,  train_docs, 'pinfo-mz-train.txt')
do_one(dev_data,    dev_docs,   'pinfo-mz-dev.txt')
do_one(dev_data,    dev_docs,   'pinfo-mz-test.txt')


# python3.6 /home/dpappas/HAR/matchzoo/main.py --phase train --model_file /home/dpappas/HAR/examples/bioasq7/config//mymodel_pinfo.config