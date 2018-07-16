import re
import sys
import json
import pickle
import random
import operator
import argparse

from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from bioasq_utils import bioclean, map_term2ind, set_unk_tokens

random.seed(1234)

def get_idf_list(tokens):
    idf_list = []
    for t in tokens:
        if t in idf:
            idf_list.append(idf[t])
        else:
            idf_list.append(max_idf)
    #
    return idf_list

def bioasq_doc_processing(doc_dict, max_doc_len):
    d_text = bioclean(doc_dict['title'] + ' ' + doc_dict['abstractText'])
    d_text_unk = set_unk_tokens(d_text, term2ind)
    d_text_inds = [term2ind[t] for t in d_text_unk][:max_doc_len]
    return d_text_inds, d_text

def bioasq_query_processing(q_text, max_q_len):
    q_text = bioclean(q_text)
    q_text_unk = set_unk_tokens(q_text, term2ind)
    #
    if len(q_text_unk) > max_q_len:
        tok_idf = []
        for token in q_text_unk:
            if token in idf:
                tok_idf.append((token, idf[token]))
            else:
                tok_idf.append((token, max_idf))
        tok_idf.sort(key=lambda tup: tup[1])
        while len(q_text_unk) > max_q_len:
            q_text_unk.remove(tok_idf[0][0])
            tok_idf.pop(0)
    #
    q_text_inds = [term2ind[t] for t in q_text_unk][:max_q_len]
    #
    return q_text_inds, q_text

# To swsto!!!
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

def get_overlap_features_mode_2(q_tokens, d_tokens, q_idf):
    # Map term to idf before set() change the term order
    q_terms_idf = {}
    for i in range(len(q_tokens)):
        q_terms_idf[q_tokens[i]] = q_idf[i]
    #
    # Query Uni and Bi gram sets
    query_uni_list = q_tokens
    query_bi_list = []
    for i in range(len(q_tokens) - 1):
        query_bi_list.append((q_tokens[i], q_tokens[i + 1]))
    #
    # Doc Uni and Bi gram sets
    doc_uni_list = []
    doc_bi_list = []
    for i in range(len(d_tokens) - 1):
        doc_uni_list.append(d_tokens[i])
        doc_bi_list.append((d_tokens[i], d_tokens[i + 1]))
    doc_uni_list.append(d_tokens[-1])
    #
    doc_uni_counter = Counter(doc_uni_list)
    doc_bi_counter = Counter(doc_bi_list)
    #
    unigram_overlap = 0
    idf_uni_overlap = 0
    idf_uni_sum = 0
    for ug in query_uni_list:
        if ug in doc_uni_counter:
            unigram_overlap += doc_uni_counter[ug]
            idf_uni_overlap += q_terms_idf[ug] * doc_uni_counter[ug]
        idf_uni_sum += q_terms_idf[ug]
    unigram_overlap /= len(query_uni_list)
    idf_uni_overlap /= idf_uni_sum
    #
    bigram_overlap = 0
    for bg in query_bi_list:
        if bg in doc_bi_list:
            bigram_overlap += doc_bi_counter[bg]
    bigram_overlap /= len(query_bi_list)
    #
    return [unigram_overlap, bigram_overlap, idf_uni_overlap]

def get_overlap_features_mode_3(q_tokens, d_tokens, q_idf):
    with open('../data/stopwords.pkl', 'rb') as f:
        stopwords = set(pickle.load(f))
    #
    # stemmer = SnowballStemmer("english")
    # q_tokens = [stemmer.stem(t) for t in q_tokens]
    # d_tokens = [stemmer.stem(t) for t in d_tokens]
    #
    # Map term to idf before set() change the term order
    q_terms_idf = {}
    for i in range(len(q_tokens)):
        q_terms_idf[q_tokens[i]] = q_idf[i]

    # Query Uni and Bi gram sets
    query_uni_list = [t for t in q_tokens if t not in stopwords]
    query_bi_list = []
    for i in range(len(q_tokens) - 1):
        query_bi_list.append((q_tokens[i], q_tokens[i + 1]))
    #
    # Doc Uni and Bi gram sets
    doc_uni_list = []
    doc_bi_list = []
    for i in range(len(d_tokens) - 1):
        if d_tokens[i] not in stopwords:
            doc_uni_list.append(d_tokens[i])
        doc_bi_list.append((d_tokens[i], d_tokens[i + 1]))
    doc_uni_list.append(d_tokens[-1])
    #
    doc_uni_counter = Counter(doc_uni_list)
    doc_bi_counter = Counter(doc_bi_list)
    #
    unigram_overlap = 0
    idf_uni_overlap = 0
    idf_uni_sum = 0
    for ug in query_uni_list:
        if ug in doc_uni_counter:
            unigram_overlap += doc_uni_counter[ug]
            idf_uni_overlap += q_terms_idf[ug] * doc_uni_counter[ug]
        idf_uni_sum += q_terms_idf[ug]
    unigram_overlap /= len(query_uni_list)
    idf_uni_overlap /= idf_uni_sum
    #
    bigram_overlap = 0
    for bg in query_bi_list:
        if bg in doc_bi_list:
            bigram_overlap += doc_bi_counter[bg]
    bigram_overlap /= len(query_bi_list)
    #
    return [unigram_overlap, bigram_overlap, idf_uni_overlap]

def get_overlap_features_mode_4(q_tokens, d_tokens, q_idf):
    with open('../data/stopwords.pkl', 'rb') as f:
        stopwords = set(pickle.load(f))
    #
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
        if d_tokens[i] not in stopwords:
            doc_uni_set.add(d_tokens[i])
        doc_bi_set.add((d_tokens[i], d_tokens[i + 1]))
    doc_uni_set.add(d_tokens[-1])
    #
    unigram_overlap = 0
    idf_uni_overlap = 0
    idf_uni_sum = 0
    query_uni_set = set([t for t in query_uni_set if t not in stopwords])
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

def produce_pos_neg_pairs(data, docset, max_year):
    pairs_list              = []
    #
    query_list              = []
    query_len_list          = []
    query_idf_list          = []
    #
    pos_doc_list            = []
    neg_doc_list            = []
    #
    pos_doc_bm25_list       = []
    neg_doc_bm25_list       = []
    #
    pos_doc_normBM25_list   = []
    neg_doc_normBM25_list   = []
    #
    pos_doc_overlap_list    = []
    neg_doc_overlap_list    = []
    #
    for q in tqdm(data['queries']):
        #
        rel_ret_set = []
        non_rel_set = []
        #
        rel_set = set(q['relevant_documents'])
        for d in q['retrieved_documents']:
            doc_id = d['doc_id']
            if doc_id in rel_set:
                rel_ret_set.append(d)
            else:
                non_rel_set.append(d)
        #
        query_inds, query_tokens = bioasq_query_processing(q['query_text'], 30)
        query_idf = get_idf_list(query_tokens)
        #
        not_found_pos = 0
        for pos_doc in rel_ret_set:
            pos_doc_id = pos_doc['doc_id']
            if pos_doc['doc_id'] not in docset:
                not_found_pos += 1
                continue
            #
            # Choose negative document published before 2016
            found = False
            tries = 0
            if non_rel_set:
                while not found and tries < len(non_rel_set):
                    neg_doc = random.choice(non_rel_set)
                    neg_doc_id = neg_doc['doc_id']
                    try:
                        pub_year = int(docset[neg_doc_id]['publicationDate'].split('-')[0])
                    except ValueError:
                        continue
                    found = (pub_year <= max_year)
                    tries += 1
            if not found:
                continue
            #
            pairs_list.append({'pos': pos_doc_id, 'neg': neg_doc_id})
            #
            pos_doc_inds, pos_doc_tokens = bioasq_doc_processing(docset[pos_doc_id], 300)
            neg_doc_inds, neg_doc_tokens = bioasq_doc_processing(docset[neg_doc_id], 300)
            #
            pos_doc_BM25 = pos_doc['bm25_score']
            neg_doc_BM25 = neg_doc['bm25_score']
            #
            pos_doc_normBM25 = pos_doc['norm_bm25_score']
            neg_doc_normBM25 = neg_doc['norm_bm25_score']
            #
            pos_doc_overlap = get_overlap_features_mode_1(query_tokens, pos_doc_tokens, query_idf)
            neg_doc_overlap = get_overlap_features_mode_1(query_tokens, neg_doc_tokens, query_idf)
            #
            query_list.append(query_inds)
            query_len_list.append(len(query_inds))
            query_idf_list.append(query_idf)
            #
            pos_doc_list.append(pos_doc_inds)
            pos_doc_bm25_list.append(pos_doc_BM25)
            pos_doc_normBM25_list.append(pos_doc_normBM25)
            pos_doc_overlap_list.append(pos_doc_overlap)
            #
            neg_doc_list.append(neg_doc_inds)
            neg_doc_bm25_list.append(neg_doc_BM25)
            neg_doc_normBM25_list.append(neg_doc_normBM25)
            neg_doc_overlap_list.append(neg_doc_overlap)
            #
        if not_found_pos > 0:
            print('{0} relevant documents are not in the docset.'.format(not_found_pos))
    #
    pairs_data = {
        'queries':              query_list,
        'queries_len':          query_len_list,
        'queries_idf':          query_idf_list,
        'pos_docs':             pos_doc_list,
        'neg_docs':             neg_doc_list,
        'pos_docs_BM25':        pos_doc_bm25_list,
        'pos_docs_normBM25':    pos_doc_normBM25_list,
        'pos_docs_overlap':     pos_doc_overlap_list,
        'neg_docs_BM25':        neg_doc_bm25_list,
        'neg_docs_normBM25':    neg_doc_normBM25_list,
        'neg_docs_overlap':     neg_doc_overlap_list,
        'pairs':                pairs_list,
        'num_pairs':            len(pairs_list)
    }
    #
    return pairs_data

def produce_reranking_inputs(data, docset, max_year):
    query_data_list = []
    for q in tqdm(data['queries']):
        query_data = {}
        #
        q_id = q['query_id']
        query_inds, query_tokens = bioasq_query_processing(q['query_text'], 30)
        query_idf = get_idf_list(query_tokens)
        #
        doc_id_list = []
        doc_list = []
        doc_BM25_list = []
        doc_norm_BM25_list = []
        doc_overlap_list = []
        for doc in q['retrieved_documents']:
            #
            # Discard document if published after 2016
            try:
                pub_year = int(docset[doc['doc_id']]['publicationDate'].split('-')[0])
            except ValueError:
                continue
            if pub_year > max_year:
                continue
            #
            doc_inds, doc_tokens = bioasq_doc_processing(docset[doc['doc_id']], 300)
            doc_BM25 = doc['bm25_score']
            doc_normBM25 = doc['norm_bm25_score']
            doc_overlap = get_overlap_features_mode_1(query_tokens, doc_tokens, query_idf)
            #
            doc_id_list.append(doc['doc_id'])
            doc_list.append(doc_inds)
            doc_BM25_list.append(doc_BM25)
            doc_norm_BM25_list.append(doc_normBM25)
            doc_overlap_list.append(doc_overlap)
            #
        query_data['id'] = q_id
        query_data['token_inds'] = query_inds
        query_data['query_len'] = len(query_inds)
        query_data['idf'] = query_idf
        query_data['retrieved_samples'] = {
            'documents':    doc_id_list,
            'doc_list':     doc_list,
            'doc_BM25':     doc_BM25_list,
            'doc_normBM25': doc_norm_BM25_list,
            'doc_overlap':  doc_overlap_list,
            'n_ret_docs':   len(doc_id_list)
        }
        #
        query_data_list.append(query_data)
        #
    return query_data_list

if __name__ == '__main__':
    topk = 100
    #
    ind         = sys.argv.index('-config')
    config_file = sys.argv[ind + 1]
    parser      = argparse.ArgumentParser()
    parser.add_argument('-config', dest='config_file')
    args        = parser.parse_args()
    print(args.config_file)
    #
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    #
    data_directory          = ''
    #
    bm25_data_path_train    = data_directory + 'data/bioasq6_bm25_top{0}/bioasq6_bm25_top{0}.train.pkl'.format(topk, topk)
    docset_path_train       = data_directory + 'data/bioasq6_bm25_top{0}/bioasq6_bm25_docset_top{0}.train.pkl'.format(topk, topk)
    #
    bm25_data_path_dev      = data_directory + 'data/bioasq6_bm25_top{0}/bioasq6_bm25_top{0}.dev.pkl'.format(topk, topk)
    docset_path_dev         = data_directory + 'data/bioasq6_bm25_top{0}/bioasq6_bm25_docset_top{0}.dev.pkl'.format(topk, topk)
    #
    w2v_path                = config['WORD_EMBEDDINGS_FILE']
    term2ind_path           = config['TERM_TO_IND']
    idf_path                = config['IDF_FILE']
    #
    with open(bm25_data_path_train, 'rb') as f:
        data_train = pickle.load(f)
    #
    with open(docset_path_train, 'rb') as f:
        docset_train = pickle.load(f)
    #
    with open(bm25_data_path_dev, 'rb') as f:
        data_dev = pickle.load(f)
    #
    with open(docset_path_dev, 'rb') as f:
        docset_dev = pickle.load(f)
    #
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    #
    print('All data loaded. Pairs generation started..')
    #
    max_idf = max(idf.items(), key=operator.itemgetter(1))[1]
    #
    term2ind = map_term2ind(w2v_path)
    with open(config['TERM_TO_IND'], 'wb') as f:
        pickle.dump(term2ind, f)
    #
    # ===================================================================
    # ===== Produce Pos/Neg pairs for the training subset of queries ====
    # ===================================================================
    print('Producing Pos-Neg pairs for training data..')
    inputs = produce_pos_neg_pairs(data_train, docset_train, max_year=2015)
    with open('data/bioasq6.top{0}.train_pairs.pkl'.format(topk), 'wb') as f:
        pickle.dump(inputs, f)
    #
    # ===================================================================
    # ==== Produce Pos/Neg pairs for the development subset of queries ==
    # ===================================================================
    print('Producing Pos-Neg pairs for dev data..')
    inputs = produce_pos_neg_pairs(data_dev, docset_dev, max_year=2016)
    with open('data/bioasq6.top{0}.dev_pairs.pkl'.format(topk), 'wb') as f:
        pickle.dump(inputs, f)
    #
    # ===================================================================
    # === Produce Reranking inputs for the training subset of queries ===
    # ===================================================================
    print('Producing reranking data for training..')
    reranking_data = produce_reranking_inputs(data_train, docset_train, max_year=2015)
    with open('data/bioasq6.top{0}.train_rerank.pkl'.format(topk), 'wb') as f:
        # This allows memory efficient reading of each query object.
        pickler = pickle.Pickler(f, protocol=2)
        for e in reranking_data:
            pickler.dump(e)
    #
    # ===================================================================
    # == Produce Reranking inputs for the development subset of queries =
    # ===================================================================
    print('Producing reranking data for dev..')
    reranking_data = produce_reranking_inputs(data_dev, docset_dev, max_year=2016)
    with open('data/bioasq6.top{0}.dev_rerank.pkl'.format(topk), 'wb') as f:
        # This allows memory efficient reading of each query object.
        pickler = pickle.Pickler(f, protocol=2)
        for e in reranking_data:
            pickler.dump(e)




