
import json, pickle, re
from collections import Counter
from elasticsearch import Elasticsearch
from pprint import pprint
from textblob import TextBlob
import nltk
from nltk import ngrams

with open('/home/dpappas/bioasq_all/stopwords.pkl', 'rb') as f:
    stopwords = pickle.load(f)

stopwords       = stopwords.union(nltk.corpus.stopwords.words("english"))
print(len(stopwords))

def keep_only_longest(phrases):
    ret = []
    for phrase in sorted(phrases, key=lambda x : len(x), reverse=True):
        if(any(phrase in t for t in ret)):
            continue
        else:
            ret.append(phrase)
    return ret

def retrieve_some_docs(qtext):
    tokenized_body  = bioclean_mod(qtext)
    tokenized_body  = [t for t in tokenized_body if t not in stopwords]
    question        = ' '.join(tokenized_body)
    bod = {
        'size' : 100,
        "query": {
            "bool" : {
                "should" : [{"match": {"section_text": {"query": question}}}] + [
                    {"match_phrase": {"section_text": {"query": chunk}}}
                    for chunk in get_noun_chunks(qtext)
                ],
                "minimum_should_match" : 1,
                "boost" : 1.0
            }
        }
    }
    res = es.search(index=index, body=bod)
    return res

def get_noun_chunks(text):
    blob    = TextBlob(text)
    pt      = blob.pos_tags
    nps     = []
    for i in range(1,5):
        nps.extend(
            [
                ' '.join([tt[0] for tt in gr])
                for gr in ngrams(pt, i)
                if(
                    all(
                        (t[1] in ['NNP','NN','JJ','ADJ','ADV'] and t[0].lower() not in stopwords)
                        for t in gr
                    )
                )
            ]
        )
    ret = list(blob.noun_phrases)+nps
    ret = keep_only_longest(ret)
    return ret

# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod = lambda t: re.sub(
    '[.,?;*!%^&_+():-\[\]{}]', '',
    t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()
).split()

index   = 'allenai_covid_index_2020_11_29_01'
es      = Elasticsearch(['127.0.0.1:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

# fpath   = 'C:\\Users\\dvpap\\Downloads\\BioASQ-taskSynergy-dryRun-testset'
fpath   = '/home/dpappas/BioASQ-taskSynergy-dryRun-testset'
d       = json.load(open(fpath))

pprint(d)

for question in d['questions']:
    qtype   = question['type']
    qtext   = question['body']
    res = retrieve_some_docs(qtext)
    # pprint(res)
    c = Counter()
    c.update(
        item['_id'].split()[0]
        for item in res['hits']['hits']
    )
    ####################################################################
    print(qtext)
    print(get_noun_chunks(qtext))
    pprint(c)
    for item in res['hits']['hits']:
        print(40 * '-')
        print(item['_id'])
        print(item['_score'])
        print(item['_source']['rank'])
        print(item['_source']['section_type'])
        print(item['_source']['section_text'])
    break





