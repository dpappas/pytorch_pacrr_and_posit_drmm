
import re, math, pickle
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from elasticsearch  import Elasticsearch
from elasticsearch.helpers import scan
import gensim

bioclean_mod = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()

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
    bod             = {"query": {"bool": {"must": [{"term": {"dataset": 'train'}}]}}}
    items           = scan(es, query=bod, index=questions_index, doc_type=questions_map)
    total           = es.count(index=questions_index, body=bod)['count']
    ################################################
    return items, total

def get_first_n(question, n):
    question    = bioclean_mod(question)
    question    = [t for t in question if t not in stopwords]
    question    = ' '.join(question)
    ################################################
    doc_index   = 'natural_questions_0_1'
    es          = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=300, max_retries=10, retry_on_timeout=True)
    ################################################
    bod = {
        "size": n,
        "query": {"match": {"paragraph_text": question}}
    }
    res = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

def load_idfs_from_df(df_path):
    print('Loading IDF tables')
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    N   = 2684631
    idf = dict(
        [
            (
                item[0],
                math.log((N*1.0) / (1.0*item[1]))
            )
            for item in df.items()
        ]
    )
    ##############
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    ##############
    print('Loaded idf tables with max idf {}'.format(max_idf))
    return idf, max_idf

def get_bm25_metrics(avgdl=0., mean=0., deviation=0.):
    if (avgdl == 0):
        total_words = 0
        total_docs  = 0
        docs, total = get_all_docs()
        for doc in tqdm(docs, total=total):
            sents = sent_tokenize(doc['_source']['paragraph_text'])
            for s in sents:
                total_words += len(tokenize(s))
                total_docs += 1.
        avgdl = float(total_words) / float(total_docs)
        print('avgdl {} computed'.format(avgdl))
    else:
        print('avgdl {} provided'.format(avgdl))
    #
    if (mean == 0 and deviation == 0):
        BM25scores = []
        k1, b = 1.2, 0.75
        quests, total = get_all_quests()
        for quest in tqdm(quests, total=total):
            qtext           = quest['_source']['question']
            if('<table>' in quest['_source']['long_answer'].lower()):
                continue
            q_toks          = tokenize(qtext)
            all_retr_docs   = get_first_n(qtext, 100)
            for retr_doc in all_retr_docs:
                sents = sent_tokenize(retr_doc['_source']['paragraph_text'])
                for sent in sents:
                    BM25score = similarity_score(q_toks, tokenize(sent), k1, b, idf, avgdl, False, 0, 0, max_idf)
                    BM25scores.append(BM25score)
        mean = sum(BM25scores) / float(len(BM25scores))
        nominator = 0
        for score in BM25scores:
            nominator += ((score - mean) ** 2)
        deviation = math.sqrt((nominator) / float(len(BM25scores) - 1))
        print('mean {} computed'.format(mean))
        print('deviation {} computed'.format(deviation))
    else:
        print('mean {} provided'.format(mean))
        print('deviation {} provided'.format(deviation))
    return avgdl, mean, deviation

df_path     = '/home/dpappas/NQ_data/NQ_my_tokenize_df.pkl'
stop_path   = '/home/dpappas/bioasq_all/bioasq_data/document_retrieval/stopwords.pkl'

with open(stop_path, 'rb') as f:
    stopwords = pickle.load(f)

idf, max_idf = load_idfs_from_df(df_path)

avgdl, mean, deviation = get_bm25_metrics(avgdl=0, mean=0, deviation=0)

print(avgdl, mean, deviation)
