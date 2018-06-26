

import json
import cPickle as pickle
from pprint import pprint
import re
import operator
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
from difflib import SequenceMatcher

UNK_TOKEN = '*UNK*'

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def similar(a, b):
    return max(
        [
            SequenceMatcher(None, a, b).ratio(),
            SequenceMatcher(None, b, a).ratio()
        ]
    )

def many_similar(one_sent, many_sents):
    return max(
        [
            similar(one_sent, s)
            for s in many_sents
        ]
    )

def map_term2ind(w2v_path):
    word_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    vocabulary  = sorted(list(word_vectors.vocab.keys()))
    term2ind = dict([t[::-1] for t in enumerate(vocabulary, start=1)])
    term2ind[UNK_TOKEN] = max(term2ind.items(), key=operator.itemgetter(1))[1] + 1	# Index of *UNK* token
    print('Size of voc: {0}'.format(len(vocabulary)))
    print('Unknown terms\'s id: {0}'.format(term2ind['*UNK*']))
    return term2ind

def set_unk_tokens(tokens, term2ind):
    clear_tokens = []
    for t in tokens:
        if t in term2ind:
            clear_tokens.append(t)
        else:
            clear_tokens.append(UNK_TOKEN)
    return clear_tokens

def first_alpha_is_upper(sent):
    specials = [
        '__EU__','__SU__','__EMS__','__SMS__','__SI__',
        '__ESB','__SSB__','__EB__','__SB__','__EI__',
        '__EA__','__SA__','__SQ__','__EQ__','__EXTLINK',
        '__XREF','__URI', '__EMAIL','__ARRAY','__TABLE',
        '__FIG','__AWID','__FUNDS'
    ]
    for special in specials:
        sent = sent.replace(special,'')
    for c in sent:
        if(c.isalpha()):
            if(c.isupper()):
                return True
            else:
                return False
    return False

def ends_with_special(sent):
    sent = sent.lower()
    ind = [item.end() for item in re.finditer('[\W\s]sp.|[\W\s]nos.|[\W\s]figs.|[\W\s]sp.[\W\s]no.|[\W\s][vols.|[\W\s]cv.|[\W\s]fig.|[\W\s]e.g.|[\W\s]et[\W\s]al.|[\W\s]i.e.|[\W\s]p.p.m.|[\W\s]cf.|[\W\s]n.a.', sent)]
    if(len(ind)==0):
        return False
    else:
        ind = max(ind)
        if (len(sent) == ind):
            return True
        else:
            return False

def split_sentences(text):
    sents = [l.strip() for l in sent_tokenize(text)]
    ret = []
    i = 0
    while (i < len(sents)):
        sent = sents[i]
        while (
            ((i + 1) < len(sents)) and
            (
                ends_with_special(sent) or
                not first_alpha_is_upper(sents[i+1].strip())
                # sent[-5:].count('.') > 1       or
                # sents[i+1][:10].count('.')>1   or
                # len(sent.split()) < 2          or
                # len(sents[i+1].split()) < 2
            )
        ):
            sent += ' ' + sents[i + 1]
            i += 1
        ret.append(sent.replace('\n',' ').strip())
        i += 1
    return ret

def get_sents(ntext):
    sents = []
    for subtext in ntext.split('\n'):
        subtext = re.sub( '\s+', ' ', subtext.replace('\n',' ') ).strip()
        if (len(subtext) > 0):
            ss = split_sentences(subtext)
            sents.extend([ s for s in ss if(len(s.strip())>0)])
    if(len(sents[-1]) == 0 ):
        sents = sents[:-1]
    return sents

def preprocess_bioasq_data(bioasq_data_path):
    data = json.load(open(bioasq_data_path, 'r'))
    ddd = {}
    for quest in data['questions']:
        if ('snippets' in quest):
            for sn in quest['snippets']:
                pmid = sn['document'].split('/')[-1]
                ttt = sn['text'].strip()
                bod = quest['body'].strip()
                if (bod not in ddd):
                    ddd[bod] = {}
                if (pmid not in ddd[bod]):
                    ddd[bod][pmid] = [ttt]
                else:
                    ddd[bod][pmid].append(ttt)
    return ddd

bioasq_data_path    = '/home/DATA/Biomedical/bioasq6/bioasq6_data/BioASQ-trainingDataset6b.json'
ddd                 = preprocess_bioasq_data(bioasq_data_path)
#
abs_path            = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.train.pkl'
all_abs             = pickle.load(open(abs_path,'rb'))
#
bm25_scores_path    = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.train.pkl'
bm25_scores         = pickle.load(open(bm25_scores_path, 'rb'))
# #
# d1      = bm25_scores['queries'][0]
# pprint(d1)

for quer in bm25_scores['queries']:
    for retr in quer['retrieved_documents']:
        doc_id      = retr['doc_id']
        doc_title   = get_sents(all_abs[doc_id]['title'])
        doc_text    = get_sents(all_abs[doc_id]['abstractText'])
        all_sents   = doc_title+doc_text
        if(retr['is_relevant']):
            relevant_snippets = []
            if(doc_id in ddd[quer['query_text']]):
                for rt in ddd[quer['query_text']][doc_id]:
                    relevant_snippets.extend(get_sents(rt))
                for s in all_sents:
                    for r in relevant_snippets:
                        similarity = similar(s,r)
                        if(similarity >= 0.8 or r in s):
                            print similarity
                            print doc_id
                            print s
                            print r
                            print 20 * '-'
        # print(retr['bm25_score'])
        # print(retr['norm_bm25_score'])
        # print(retr['is_relevant'])
    # break







