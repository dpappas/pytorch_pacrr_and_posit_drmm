

from nltk.tokenize import sent_tokenize
import cPickle as pickle
from nltk.corpus import stopwords
import numpy as np
import re
import operator
from gensim.models import KeyedVectors

# idf_path        = '/home/DATA/Biomedical/bioasq6/bioasq6_data/IDF.pkl'
idf_path        = '/home/dpappas/IDF_python_v2.pkl'
stopw_path      = '/home/DATA/Biomedical/other/BiomedicalWordEmbeddings/stopwords.txt'

bioclean        = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
stopwords1      = list([t.strip() for t in open(stopw_path).readlines()])
stopwords2      = list(stopwords.words('english'))
stop            = set(stopwords1 + stopwords2)
unk_tok         = 'UNKN'
idf             = pickle.load(open(idf_path, 'rb'))
max_idf         = max(idf.items(), key=operator.itemgetter(1))[1]

def get_idf_list(tokens):
    idf_list = []
    for t in tokens:
        if t in idf:
            idf_list.append(idf[t])
        else:
            idf_list.append(max_idf)
    #
    return idf_list

def get_overlap_features_mode_1(q_tokens, d_tokens):
    q_idf       = get_idf_list(q_tokens)
    # Map term to idf before set() change the term order
    q_terms_idf = {}
    for i in range(len(q_tokens)):
        q_terms_idf[q_tokens[i]] = q_idf[i]
    #
    # Query Uni and Bi gram sets
    query_uni_set = set()
    query_bi_set = set()
    for i in range(len(q_tokens) - 1):
        query_uni_set.add(q_tokens[i])
        query_bi_set.add((q_tokens[i], q_tokens[i + 1]))
    query_uni_set.add(q_tokens[-1])
    #
    # Doc Uni and Bi gram sets
    doc_uni_set = set()
    doc_bi_set = set()
    for i in range(len(d_tokens) - 1):
        doc_uni_set.add(d_tokens[i])
        doc_bi_set.add((d_tokens[i], d_tokens[i + 1]))
    doc_uni_set.add(d_tokens[-1])
    #
    unigram_overlap = 0
    idf_uni_overlap = 0
    idf_uni_sum = 0
    for ug in query_uni_set:
        if ug in doc_uni_set:
            unigram_overlap += 1
            idf_uni_overlap += q_terms_idf[ug]
        idf_uni_sum += q_terms_idf[ug]
    unigram_overlap /= len(query_uni_set)
    idf_uni_overlap /= idf_uni_sum
    #
    bigram_overlap = 0
    for bg in query_bi_set:
        if bg in doc_bi_set:
            bigram_overlap += 1
    bigram_overlap /= len(query_bi_set)
    #
    return [unigram_overlap, bigram_overlap, idf_uni_overlap]

def get_index(token, t2i):
    try:
        return t2i[token]
    except KeyError:
        return t2i[unk_tok]

def get_sim_mat(stoks, qtoks):
    sm = np.zeros((len(stoks), len(qtoks)))
    for i in range(len(qtoks)):
        for j in range(len(stoks)):
            if(qtoks[i] == stoks[j]):
                sm[j,i] = 1.
    return sm

def remove_stopw(tokens):
    return [
        tok if( tok.lower() not in stop) else unk_tok
        for tok in tokens
    ]

def get_item_inds(item, question, t2i, remove_stopwords=False):
    passage         = item['title'] + ' ' + item['abstractText']
    all_sims        = get_sim_mat(bioclean(passage), bioclean(question))
    passage_toks    = bioclean(passage)
    question_toks   = bioclean(question)
    if(remove_stopwords):
        passage_toks    = remove_stopw(passage_toks)
        question_toks   = remove_stopw(question_toks)
    sents_inds          = [get_index(token, t2i) for token in passage_toks]
    quest_inds          = [get_index(token, t2i) for token in question_toks]
    additional_features = get_overlap_features_mode_1(bioclean(question_toks), passage_toks)
    return sents_inds, quest_inds, all_sims, additional_features

def text2indices(text, t2i):
    return [get_index(token, t2i) for token in bioclean(text)]




