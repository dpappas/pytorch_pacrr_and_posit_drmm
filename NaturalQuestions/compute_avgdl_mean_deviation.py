
import re, math
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from elasticsearch  import Elasticsearch
from elasticsearch.helpers import scan

def clean_start_end(word):
    word = re.sub(r'(^\W+)', r'\1 ', word)
    word = re.sub(r'(\W+$)', r' \1', word)
    word = re.sub(r'\s+', ' ', word)
    return word.strip()

def tokenize(text):
    ret = []
    for token in word_tokenize(text):
        ret.extend(clean_start_end(token).split())
    return ret

def tf(term, document):
    tf = 0
    for word in document:
        if word == term:
            tf += 1
    if len(document) == 0:
        return tf
    else:
        return tf / len(document)

def similarity_score(query, document, k1, b, idf_scores, avgdl, normalize, mean, deviation, rare_word):
    score = 0
    for query_term in query:
        if query_term not in idf_scores:
            score += rare_word * (
                    (tf(query_term, document) * (k1 + 1)) /
                    (
                            tf(query_term, document) +
                            k1 * (1 - b + b * (len(document) / avgdl))
                    )
            )
        else:
            score += idf_scores[query_term] * ((tf(query_term, document) * (k1 + 1)) / (
                        tf(query_term, document) + k1 * (1 - b + b * (len(document) / avgdl))))
    if normalize:
        return ((score - mean) / deviation)
    else:
        return score

def compute_avgdl(documents):
    total_words = 0
    for document in documents:
        total_words += len(document)
    avgdl = total_words / len(documents)
    return avgdl

def get_all_docs():
    ################################################
    doc_index   = 'natural_questions_0_1'
    doc_map     = "natural_questions_map_0_1"
    es          = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    items       = scan(es, query=None, index=doc_index, doc_type=doc_map)
    total       = es.count(index=doc_index)['count']
    ################################################
    return items, total

def get_all_quests():
    ################################################
    questions_index = 'natural_questions_q_0_1'
    questions_map   = "natural_questions_q_map_0_1"
    es              = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    items           = scan(es, query=None, index=questions_index, doc_type=questions_map)
    total           = es.count(index=questions_index)['count']
    ################################################
    return items, total

def get_bm25_metrics(avgdl=0., mean=0., deviation=0.):
    if (avgdl == 0):
        total_words = 0
        total_docs  = 0
        docs, total = get_all_docs()
        for doc in tqdm(docs, ascii=True):
            sents = sent_tokenize(doc['_source']['paragraph_text'])
            for s in sents:
                total_words += len(tokenize(s))
                total_docs += 1.
        avgdl = float(total_words) / float(total_docs)
    else:
        print('avgdl {} provided'.format(avgdl))
    #
    if (mean == 0 and deviation == 0):
        BM25scores = []
        k1, b = 1.2, 0.75
        not_found = 0
        for qid in tqdm(bioasq6_data, ascii=True):
            qtext           = bioasq6_data[qid]['body']
            all_retr_ids    = [link.split('/')[-1] for link in bioasq6_data[qid]['documents']]
            for dic in all_retr_ids:
                try:
                    sents = sent_tokenize(train_docs[dic]['title']) + sent_tokenize(train_docs[dic]['abstractText'])
                    q_toks = tokenize(qtext)
                    for sent in sents:
                        BM25score = similarity_score(q_toks, tokenize(sent), k1, b, idf, avgdl, False, 0, 0, max_idf)
                        BM25scores.append(BM25score)
                except KeyError:
                    not_found += 1
        #
        mean = sum(BM25scores) / float(len(BM25scores))
        nominator = 0
        for score in BM25scores:
            nominator += ((score - mean) ** 2)
        deviation = math.sqrt((nominator) / float(len(BM25scores) - 1))
    else:
        print('mean {} provided'.format(mean))
        print('deviation {} provided'.format(deviation))
    return avgdl, mean, deviation

avgdl, mean, deviation = get_bm25_metrics()
