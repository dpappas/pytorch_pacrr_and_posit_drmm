

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import re
import operator
from gensim.models import KeyedVectors

# w2v_path        = '/home/DATA/Biomedical/other/BiomedicalWordEmbeddings/binary/biomedical-vectors-200.bin'
bioclean        = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
stopwords1      = list([t.strip() for t in open('/home/DATA/Biomedical/other/BiomedicalWordEmbeddings/stopwords.txt').readlines()])
stopwords2      = list(stopwords.words('english'))
stop            = set(stopwords1 + stopwords2)

def get_overlap_features_mode_1(q_tokens, d_tokens, q_idf):
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
        return t2i['UNKN']

def get_sim_mat(stoks, qtoks):
    sm = np.zeros((len(stoks), len(qtoks)))
    for i in range(len(qtoks)):
        for j in range(len(stoks)):
            if(qtoks[i] == stoks[j]):
                sm[j,i] = 1.
    return sm

def get_item_inds(item, question, t2i):
    passage     = item['title'] + ' ' + item['abstractText']
    all_sims    = get_sim_mat(bioclean(passage), bioclean(question))
    sents_inds  = [get_index(token, t2i) for token in bioclean(passage)]
    quest_inds  = [get_index(token, t2i) for token in bioclean(question)]
    return sents_inds, quest_inds, all_sims

def remove_stopwords(tokens):
    return [
        tok if() else 'UNKN'
        for tok in tokens
    ]


