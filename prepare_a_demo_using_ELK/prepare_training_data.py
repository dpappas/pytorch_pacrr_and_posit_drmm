
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
nlp = spacy.load("en_core_sci_md")
# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)
linker = UmlsEntityLinker(resolve_abbreviations=True)
nlp.add_pipe(linker)

################################################################################

import  json
from    pprint import pprint
import  nltk
import  re
from elasticsearch import Elasticsearch
from tqdm import tqdm
import pickle

################################################################################

# recall: 0.554446184347
def get_first_n_1(question_tokens, n):
    question    = ' '.join(question_tokens)
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {"match": {"AbstractText": question}},
                    {"match": {"ArticleTitle": question}},
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and"
                        }
                    }
                ],
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.516323587896
def get_first_n_2(question_tokens, n):
    question    = ' '.join(question_tokens)
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {"match": {"AbstractText": question}},
                    {"match": {"ArticleTitle": question}},
                    # {
                    #     "multi_match": {
                    #         "query": question,
                    #         "type": "most_fields",
                    #         "fields": ["AbstractText", "ArticleTitle"],
                    #         "operator": "and"
                    #     }
                    # }
                ],
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.571223691706
def get_first_n_3(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.542825659381
def get_first_n_4(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    # {
                    #     "multi_match": {
                    #         "query": question,
                    #         "type": "most_fields",
                    #         "fields": ["AbstractText", "ArticleTitle"],
                    #         "operator": "and",
                    #         "boost": sum(idf_scores)
                    #     }
                    # }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.572105425712
def get_first_n_5(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.562218834681
def get_first_n_6(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"ArticleTitle": {"query": q_tok, "boost": idf_score}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.567039063307
def get_first_n_7(question_tokens, n):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok in question_tokens:
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": 1}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {"match": {"AbstractText": {"query": question, "boost": 1}}},
                    {"match": {"ArticleTitle": {"query": question, "boost": 1}}},
                    {"multi_match": {"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "operator": "and", "boost": 1}},
                    {"multi_match": {"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "minimum_should_match": "50%"}}
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.572181183288
def get_first_n_8(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.571551326636
def get_first_n_9(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    idf_scores  = [idf/float(max(idf_scores)) for idf in idf_scores]
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.560685592131
def get_first_n_10(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    idf_scores  = [idf/float(max_idf) for idf in idf_scores]
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2017",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.574698935197
def get_first_n_11(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2016",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.515151622758
def get_first_n_12(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2015",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.57380594438
def get_first_n_13(question_tokens, n, idf_scores):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"Chemicals.NameOfSubstance": {"query": q_tok,"boost": idf_score}}})
        the_shoulds.append({"match": {"MeshHeadings.text": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"SupplMeshList.text": {"query": q_tok,"boost": idf_score}}})
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2016",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.5816116442796173
def get_first_n_14(question_tokens, n, idf_scores, entities, abbreviations):
    question    = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"Chemicals.NameOfSubstance": {"query": q_tok,"boost": idf_score}}})
        the_shoulds.append({"match": {"MeshHeadings.text": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"SupplMeshList.text": {"query": q_tok,"boost": idf_score}}})
    ################################################
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    for phrase in entities+abbreviations:
        # print("|{}|".format(phrase))
        the_shoulds.append({"match_phrase": {"AbstractText": {"query": phrase, "boost": sum(idf_scores)}}})
        the_shoulds.append({"match_phrase": {"Chemicals.NameOfSubstance": {"query": phrase, "boost": sum(idf_scores)}}})
        the_shoulds.append({"match_phrase": {"MeshHeadings.text": {"query": phrase, "boost": sum(idf_scores)}}})
        the_shoulds.append({"match_phrase": {"SupplMeshList.text": {"query": phrase, "boost": sum(idf_scores)}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2016",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.5915700385340271
def get_first_n_15(question_tokens, n, idf_scores, entities, abbreviations):
    question = ' '.join(entities + abbreviations)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"Chemicals.NameOfSubstance": {"query": q_tok,"boost": idf_score}}})
        the_shoulds.append({"match": {"MeshHeadings.text": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"SupplMeshList.text": {"query": q_tok,"boost": idf_score}}})
    ################################################
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    for phrase in entities+abbreviations:
        # print("|{}|".format(phrase))
        the_shoulds.append({"match_phrase": {"AbstractText": {"query": phrase, "boost": sum(idf_scores)}}})
        the_shoulds.append({"match_phrase": {"Chemicals.NameOfSubstance": {"query": phrase, "boost": sum(idf_scores)}}})
        the_shoulds.append({"match_phrase": {"MeshHeadings.text": {"query": phrase, "boost": sum(idf_scores)}}})
        the_shoulds.append({"match_phrase": {"SupplMeshList.text": {"query": phrase, "boost": sum(idf_scores)}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2016",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.5995573845213731
def get_first_n_16(question_tokens, n, idf_scores, entities, abbreviations):
    if(len(entities+abbreviations)>1):
        question = ' '.join(entities + abbreviations)
    else:
        question = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"Chemicals.NameOfSubstance": {"query": q_tok,"boost": idf_score}}})
        the_shoulds.append({"match": {"MeshHeadings.text": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"SupplMeshList.text": {"query": q_tok,"boost": idf_score}}})
    ################################################
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    for phrase in entities+abbreviations:
        # print("|{}|".format(phrase))
        the_shoulds.append({"match_phrase": {"AbstractText": {"query": phrase, "boost": sum(idf_scores)}}})
        the_shoulds.append({"match_phrase": {"Chemicals.NameOfSubstance": {"query": phrase, "boost": sum(idf_scores)}}})
        the_shoulds.append({"match_phrase": {"MeshHeadings.text": {"query": phrase, "boost": sum(idf_scores)}}})
        the_shoulds.append({"match_phrase": {"SupplMeshList.text": {"query": phrase, "boost": sum(idf_scores)}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2016",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.6061334920933832
def get_first_n_17(question_tokens, n, idf_scores, entities, abbreviations):
    if(len(entities+abbreviations)>1):
        question = ' '.join(entities + abbreviations)
    else:
        question = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"Chemicals.NameOfSubstance": {"query": q_tok,"boost": idf_score}}})
        the_shoulds.append({"match": {"MeshHeadings.text": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"SupplMeshList.text": {"query": q_tok,"boost": idf_score}}})
    ################################################
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    for phrase in entities+abbreviations:
        # print("|{}|".format(phrase))
        idf_score =  sum([idf_val(t, idf, max_idf) for t in phrase.lower().split()])
        the_shoulds.append({"match_phrase": {"AbstractText": {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"Chemicals.NameOfSubstance": {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"MeshHeadings.text": {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"SupplMeshList.text": {"query": phrase, "boost": idf_score}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [
                    {
                        "range": {
                            "DateCompleted": {
                                "gte": "1800",
                                "lte": "2016",
                                "format": "dd/MM/yyyy||yyyy"
                            }
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "AbstractText": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "match": {
                            "ArticleTitle": {
                                "query": question,
                                "boost": sum(idf_scores)
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": question,
                            "type": "most_fields",
                            "fields": ["AbstractText", "ArticleTitle"],
                            "operator": "and",
                            "boost": sum(idf_scores)
                        }
                    },
                   {
                       "multi_match": {
                           "query"                : question,
                           "type"                 : "most_fields",
                           "fields"               : ["AbstractText", "ArticleTitle"],
                           "minimum_should_match" : "50%"
                       }
                   }
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

# recall: 0.6066111686889551
def get_first_n_18(question_tokens, n, idf_scores, entities, abbreviations):
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

# recall: 0.5903940829779357
def get_first_n_19(question_tokens, n, idf_scores, entities, abbreviations):
    if(len(entities+abbreviations)>1):
        question = ' '.join(entities + abbreviations)
    else:
        question = ' '.join(question_tokens)
    ################################################
    the_shoulds = []
    for q_tok, idf_score in zip(question_tokens, idf_scores):
        the_shoulds.append({"match": {"AbstractText": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"Chemicals.NameOfSubstance": {"query": q_tok,"boost": idf_score}}})
        the_shoulds.append({"match": {"MeshHeadings.text": {"query": q_tok, "boost": idf_score}}})
        the_shoulds.append({"match": {"SupplMeshList.text": {"query": q_tok,"boost": idf_score}}})
    ################################################
    if(len(question_tokens) > 1):
        the_shoulds.append(
            {
                "span_near": {
                    "clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens],
                    "slop": 5,
                    "in_order": False
                }
            }
        )
    ################################################
    for phrase in entities+abbreviations:
        # print("|{}|".format(phrase))
        idf_score =  sum([idf_val(t, idf, max_idf) for t in phrase.lower().split()])
        the_shoulds.append({"match_phrase": {"ArticleTitle": {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"AbstractText": {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"Chemicals.NameOfSubstance": {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"MeshHeadings.text": {"query": phrase, "boost": idf_score}}})
        the_shoulds.append({"match_phrase": {"SupplMeshList.text": {"query": phrase, "boost": idf_score}}})
    ################################################
    bod         = {
        "size": n,
        "query": {
            "bool" : {
                "must": [{"range":{"DateCompleted": {"gte": "1800", "lte": "2016", "format": "dd/MM/yyyy||yyyy"}}}],
                "should": [
                    {"match":{"AbstractText": {"query": question, "boost": sum(idf_scores)}}},
                    {"match":{"ArticleTitle": {"query": question, "boost": sum(idf_scores)}}},
                    {"multi_match":{"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"],"operator": "and", "boost": sum(idf_scores)}},
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

# recall: 0.5820400178593594
def get_first_n_21(question_tokens, n, idf_scores, entities, abbreviations):
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
        the_shoulds.append({"span_near": {"clauses": [{"span_term": {"AbstractText": w}} for w in question_tokens], "slop": 5, "boost" : sum(idf_scores), "in_order": False}})
        the_shoulds.append({"span_near": {"clauses": [{"span_term": {"ArticleTitle": w}} for w in question_tokens], "slop": 5, "boost" : sum(idf_scores), "in_order": False}})
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
                    {"multi_match":{"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "operator": "and",             "boost": sum(idf_scores)}},
                    {"multi_match":{"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "minimum_should_match": "30%", "boost": sum(idf_scores)}},
                    {"multi_match":{"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "minimum_should_match": "50%", "boost": sum(idf_scores)}},
                    {"multi_match":{"query": question, "type": "most_fields", "fields": ["AbstractText", "ArticleTitle"], "minimum_should_match": "75%", "boost": sum(idf_scores)}},
                ]+the_shoulds,
                "minimum_should_match": 1,
            }
        }
    }
    # pprint(bod)
    res         = es.search(index=doc_index, body=bod, request_timeout=120)
    return res['hits']['hits']

################################################################################

def get_scispacy(qtext):
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

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

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

################################################################################

bioclean_mod    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()).split()

stopwords       = nltk.corpus.stopwords.words("english")

################################################################################

doc_index = 'pubmed_abstracts_0_1'
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

################################################################################

fpath           = '/home/dpappas/bioasq_all/bioasq7/data/trainining7b.json'
idf_pickle_path = '/home/dpappas/bioasq_all/idf.pkl'

idf, max_idf    = load_idfs(idf_pickle_path)
training_data   = json.load(open(fpath))

################################################################################

bm25_top100_train_pkl   = {'queries': []}
bm25_docset_train_pkl   = {}
bm25_top100_dev_pkl     = {'queries': []}
bm25_docset_dev_pkl     = {}

################################################################################

fetch_no        = 100
counter         = 0
verbose         = False
for question in tqdm(training_data['questions']):
    ########################################
    abbreviations, entities = get_scispacy(question['body'])
    ########################################
    qtext       = bioclean_mod(question['body'])
    qtext       = [t for t in qtext if t not in stopwords]
    ########################################
    idf_scores  = [idf_val(w, idf, max_idf) for w in qtext]
    ########################################
    top100_datum = {
        'num_rel'               : len(question['documents']),
        'num_rel_ret'           : 0,
        'num_ret'               : 0,
        'query_id'              : question['id'],
        'query_text'            : question['body'],
        'relevant_documents'    : [d.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for d in question['documents']],
        'retrieved_documents'   : []
    }
    ########################################
    rank = 0
    for retr_doc in get_first_n_20(qtext, fetch_no, idf_scores, entities, abbreviations):
        rank += 1
        retr_doc = {
            'bm25_score'        : retr_doc['_score'],
            'doc_id'            : retr_doc['_source']['pmid'],
            'is_relevant'       : retr_doc['_source']['pmid'] in top100_datum['relevant_documents'],
            'norm_bm25_score'   : -1,
            'rank'              : rank
        }
        top100_datum['retrieved_documents'].append(retr_doc)
        top100_datum['num_ret'] += 1
        if(retr_doc['is_relevant']):
            top100_datum['num_rel_ret'] += 1
        bm25_docset_train_pkl[retr_doc['_source']['pmid']] = {
            'abstractText'      : retr_doc['_source']['AbstractText'],
            'title'             : retr_doc['_source']['ArticleTitle'],
            'pmid'              : retr_doc['_source']['pmid'],
            'publicationDate'   : retr_doc['_source']['DateCompleted'],
        }
    bm25_top100_train_pkl['queries'].append(top100_datum)
    ########################################
    counter += 1
    if(counter == 200):
        break

################################################################################

'''
fetch_no        = 100
recalls         = []
counter         = 0
verbose         = False
for question in tqdm(training_data['questions']):
    ########################################
    abbreviations, entities = get_scispacy(question['body'])
    ########################################
    qtext       = bioclean_mod(question['body'])
    qtext       = [t for t in qtext if t not in stopwords]
    ########################################
    idf_scores  = [idf_val(w, idf, max_idf) for w in qtext]
    ########################################
    top100_datum = {
        'num_rel'               : -1,
        'num_rel_ret'           : len(question['documents']),
        'num_ret'               : fetch_no,
        'query_id'              : question['id'],
        'query_text'            : question['body'],
        'relevant_documents'    : [d.replace('http://www.ncbi.nlm.nih.gov/pubmed/', '') for d in question['documents']],
        'retrieved_documents'   : []
    }
    ########################################
    retrieved_pmids     = []
    rank = 0
    for retr_doc in get_first_n_20(qtext, fetch_no, idf_scores, entities, abbreviations):
        rank += 1
        retrieved_pmids.append(u'http://www.ncbi.nlm.nih.gov/pubmed/{}'.format(retr_doc['_source']['pmid']))
        top100_datum['retrieved_documents'].append(
            {
                'bm25_score'        : retr_doc['_score'],
                'doc_id'            : retr_doc['_source']['pmid'],
                'is_relevant'       : retr_doc['_source']['pmid'] in top100_datum['relevant_documents'],
                'norm_bm25_score'   : -1,
                'rank'              : rank
            }
        )
        bm25_docset_train_pkl[retr_doc['_source']['pmid']] = {
            'abstractText'      : retr_doc['_source']['AbstractText'],
            'title'             : retr_doc['_source']['ArticleTitle'],
            'pmid'              : retr_doc['_source']['pmid'],
            'publicationDate'   : retr_doc['_source']['DateCompleted'],
        }
    ########################################
    recall = float(len(set(question['documents']).intersection(set(retrieved_pmids)))) / float(len(question['documents']))
    recalls.append(recall)
    ########################################
    if(verbose):
        if(recall<1):
            print(20 * '=')
            print(recall)
            print(question['body'])
            print('ENTITIES:')
            print(entities)
            print(qtext)
            print(idf_scores)
            print('SHOULD FETCH:')
            pprint(set(question['documents']) - set(retrieved_pmids))
            print(recall)
            print('SHOULD NOT FETCH:')
            pprint(list(set(retrieved_pmids) - set(question['documents']))[:5])
    ########################################
    counter += 1
    if(counter == 200):
        break

print(sum(recalls) / float(len(recalls)))
'''

################################################################################

'''
GET /pubmed_abstracts_0_1/_explain/0
{
      "query" : {
        "match" : { "message" : "elasticsearch" }
      }
}

'''

'''
{'num_rel': 9,
 'num_rel_ret': 6,
 'num_ret': 100,
 'query_id': '55031181e9bde69634000014',
 'query_text': 'Is Hirschsprung disease a mendelian or a multifactorial disorder?',
 'relevant_documents': ['12239580', '15617541', ... ],
 'retrieved_documents': [{'bm25_score': 7.02374051,
                          'doc_id': '15617541',
                          'is_relevant': True,
                          'norm_bm25_score': 3.7238850702205664,
                          'rank': 1},
                          ...

'''




