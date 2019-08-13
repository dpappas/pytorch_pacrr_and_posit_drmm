
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
nlp = spacy.load("en_core_sci_md")
# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
linker = UmlsEntityLinker(resolve_abbreviations=True)
nlp.add_pipe(linker)

import pickle, math, re, json, os
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from elasticsearch.helpers import scan
from elasticsearch import Elasticsearch

bioclean_mod    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()
bioclean        = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

doc_index = 'pubmed_abstracts_0_1'
doc_type = "abstract_map_0_1"
es = Elasticsearch(
    hosts            = [
        'palomar.ilsp.gr:9201', # palomar
        '192.168.188.86:9200', # judgment
        '192.168.188.95:9200', # harvester1
        '192.168.188.108:9200', # bionlp4
        '192.168.188.109:9200', # bionlp5
        '192.168.188.110:9200', # bionlp6
        # INGESTORS
        # '192.168.188.101:9200', # harvester3
        # '192.168.188.102:9200', # harvester4
        # '192.168.188.105:9200', # bionlp1
        # '192.168.188.106:9200', # bionlp2
        # '192.168.188.107:9200', # bionlp3
    ],
    verify_certs     = True,
    timeout          = 150,
    max_retries      = 10,
    retry_on_timeout = True
)

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

# recall: 0.6123646029503693
def get_first_n_20(question_tokens, n, idf_scores, entities, abbreviations):
    if(len(entities+abbreviations)>1):
        question = ' '.join(entities + abbreviations)
    else:
        question = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText"                : {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"Chemicals.NameOfSubstance"   : {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"MeshHeadings.text"           : {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"SupplMeshList.text"          : {"query": q_tok, "boost": idf_score}}})
        ################################################
        the_shoulds.append({"terms": {"AbstractText"                : [q_tok], "boost": idf_score}})
        the_shoulds.append({"terms": {"Chemicals.NameOfSubstance"   : [q_tok], "boost": idf_score}})
        the_shoulds.append({"terms": {"MeshHeadings.text"           : [q_tok], "boost": idf_score}})
        the_shoulds.append({"terms": {"AbstractText"                : [q_tok], "boost": idf_score}})
    ################################################
    if(len(question_tokens) > 1):
        the_shoulds.append({"span_near": {"clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens], "slop": 5, "in_order": False}})
    ################################################
    for phrase in entities+abbreviations:
        # print("|{}|".format(phrase))
        idf_score =  sum([idf_val(t, idf, max_idf) for t in phrase.lower().split()])
        the_shoulds.append({"match_phrase": {"AbstractText"                 : {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"Chemicals.NameOfSubstance"    : {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"MeshHeadings.text"            : {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"SupplMeshList.text"           : {"query": phrase, "boost": idf_score}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool": {
                "must": [{"range":{"DateCompleted": {"gte": "1800", "lte": "2016", "format": "dd/MM/yyyy||yyyy"}}}],
                "should": [
                    {"match":{"AbstractText": {"query": question, "boost": sum(idf_scores)}}},
                    {"match":{"ArticleTitle": {"query": question, "boost": sum(idf_scores)}}},
                    {"multi_match":{"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "operator": "and", "boost": sum(idf_scores)}},
                    {"multi_match":{"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "minimum_should_match": "30%"}},
                    {"multi_match":{"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "minimum_should_match": "50%"}},
                    {"multi_match":{"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "minimum_should_match": "75%"}},
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

def tokenize(text):
    return bioclean(text)

def load_idfs(idf_path):
    print('Loading IDF tables')
    ###############################
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    print('Loaded idf tables with max idf {}'.format(max_idf))
    ###############################
    return idf, max_idf

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

def get_scispacy(qtext):
    try:
        doc             = nlp(qtext)
        abbreviations   = []
        for abrv in doc._.abbreviations:
            abbreviations.append(abrv)
            abbreviations.append(abrv._.long_form)
        #
        entities        = list(doc.ents)
        entities        = [str(ent) for ent in entities]
        abbreviations   = [str(abr) for abr in abbreviations]
        return abbreviations, entities
    except:
        return [], []

def get_bm25_metrics(avgdl=0., mean=0., deviation=0.):
    ######################################################
    if (avgdl == 0):
        total_words = 0
        total_docs  = 0
        docs = scan(es, query=None, index=doc_index, doc_type=doc_type)
        for doc in tqdm(docs, total=30000000):
            sents = []
            if('ArticleTitle' in doc['_source']):
                sents += sent_tokenize(doc['_source']['ArticleTitle'])
            if('AbstractText' in doc['_source']):
                sents += sent_tokenize(doc['_source']['AbstractText'])
            for s in sents:
                total_words += len(tokenize(s))
                total_docs += 1.
        avgdl = float(total_words) / float(total_docs)
        print('avgdl {} computed'.format(avgdl))
    else:
        print('avgdl {} provided'.format(avgdl))
    ######################################################
    if (mean == 0 and deviation == 0):
        BM25scores = []
        k1, b = 1.2, 0.75
        for question in tqdm(training_data['questions']):
            qtext                   = question['body']
            q_toks                  = tokenize(qtext)
            idf_scores              = [idf_val(w, idf, max_idf) for w in q_toks]
            abbreviations, entities = get_scispacy(question['body'])
            #
            for retr_doc in get_first_n_20(qtext, 100, idf_scores, entities, abbreviations):
                sents = []
                if('ArticleTitle' in retr_doc['_source']):
                    sents += sent_tokenize(retr_doc['_source']['ArticleTitle'])
                if('AbstractText' in retr_doc['_source']):
                    sents += sent_tokenize(retr_doc['_source']['AbstractText'])
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

idf_pickle_path         = '/home/dpappas/bioasq_all/idf.pkl'
idf, max_idf            = load_idfs(idf_pickle_path)

fpath                   = '/home/dpappas/bioasq_all/bioasq7/data/trainining7b.json'
training_data           = json.load(open(fpath))

avgdl, mean, deviation  = get_bm25_metrics(avgdl=20.47079583909152, mean=0.6158675062087192, deviation=1.205199607538813)

print(avgdl, mean, deviation)




