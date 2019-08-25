
import os, sys, json, pickle, random, re
from pprint import pprint
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

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

def train_data_step1(les_data):
    ret = []
    for dato in tqdm(les_data['queries'], ascii=True):
        quest = dato['query_text']
        quest_id = dato['query_id']
        ret_pmids = [t[u'doc_id'] for t in dato[u'retrieved_documents']]
        good_pmids = [t for t in ret_pmids if t in dato[u'relevant_documents']]
        bad_pmids = [t for t in ret_pmids if t not in dato[u'relevant_documents']]
        if (len(bad_pmids) > 0):
            for gid in good_pmids:
                bid = random.choice(bad_pmids)
                ret.append((quest, quest_id, gid, bid))
    print('')
    return ret

def snip_is_relevant(one_sent, gold_snips):
    return int(
        any(
            [
                (one_sent.encode('ascii', 'ignore') in gold_snip.encode('ascii', 'ignore'))
                or
                (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii', 'ignore'))
                for gold_snip in gold_snips
            ]
        )
    )

def prep_data(the_doc, good_snips):
    good_sents          = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
    held_out_sents, good_sent_tags = [], []
    for good_text in good_sents:
        held_out_sents.append(good_text)
        good_sent_tags.append(snip_is_relevant(' '.join(bioclean(good_text)), good_snips))
    ####
    return {
        'sent_tags'     : good_sent_tags,
        'held_out_sents': held_out_sents
    }

def get_snips(quest_id, gid, bioasq6_data):
    good_snips = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            if (sn['document'].endswith(gid)):
                good_snips.extend(sent_tokenize(sn['text']))
    return good_snips

def train_data_step2(instances, docs, bioasq6_data, use_sent_tokenizer):
    for quest_text, quest_id, gid, bid in instances:
        ####################
        good_snips          = get_snips(quest_id, gid, bioasq6_data)
        good_snips          = [' '.join(bioclean(sn)) for sn in good_snips]
        ####################
        datum               = prep_data(docs[gid], good_snips)
        good_sent_tags      = datum['sent_tags']
        good_held_out_sents = datum['held_out_sents']
        #
        if (use_sent_tokenizer == False or sum(good_sent_tags) > 0):
            yield {
                'good_sent_tags'        : good_sent_tags,
                'good_held_out_sents'   : good_held_out_sents,
                'quest_text'            : quest_text
            }

def get_data_for_fastbert(bioasq6_data, use_sent_tokenizer, les_data, les_docs):
    instances = train_data_step1(les_data)
    random.shuffle(instances)
    #
    ret = []
    pbar = tqdm(iterable=train_data_step2(instances, les_docs, bioasq6_data, use_sent_tokenizer), total=14288, ascii=True)
    for datum in pbar:
        for sent, tag in zip(datum['good_held_out_sents'], datum['good_sent_tags']):
            label = 'neg' if tag == 0 else 'pos'
            line = [datum['quest_text'].replace(',', '').replace('\n', ' ') + ' ### ' + sent.replace(',', '').replace('\n', ' '), label]
            ret.append(line)
    return ret

odir = '/home/dpappas/fast_bert_models/snippet_extraction/'
if(not os.path.exists(odir)):
    os.makedirs(odir)

with open(os.path.join(odir, 'labels.csv'), 'w') as f:
    f.write('pos')
    f.write('\n')
    f.write('neg')
    f.close()

dataloc = '/home/dpappas/bioasq_all/bioasq7_data/'
(dev_data, dev_docs, train_data, train_docs, bioasq6_data) = load_all_data(dataloc=dataloc)

train_data  = get_data_for_fastbert(bioasq6_data, True, train_data, train_docs)
dev_data    = get_data_for_fastbert(bioasq6_data, True, dev_data, dev_docs)

pos_train   = [ l for l in train_data if(l[-1] == 'pos')]
random.shuffle(pos_train)
neg_train   = [ l for l in train_data if(l[-1] == 'neg')]
random.shuffle(neg_train)

lines = pos_train + neg_train[:len(pos_train)]
random.shuffle(lines)
index = 0
with open(os.path.join(odir, 'train.csv'), 'w') as f:
    f.write(','.join(['index', 'text', 'label']))
    f.write('\n')
    for line in lines:
        f.write(','.join([str(index)] + line))
        f.write('\n')
        index += 1
    f.close()

index = 0
lines = dev_data
with open(os.path.join(odir, 'val.csv'), 'w') as f:
    f.write(','.join(['index', 'text', 'label']))
    f.write('\n')
    for line in lines:
        f.write(','.join([str(index)] + line))
        f.write('\n')
        index += 1
    f.close()

