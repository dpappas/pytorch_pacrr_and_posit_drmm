import pickle, io, ijson, json
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from pymongo import MongoClient


def create_docset(docs_needed):
    print("Retrieving text for {0} documents".format(len(docs_needed)))
    docset = {}
    client = MongoClient('localhost', 27017)
    db = client.pubmedBaseline2018
    collection = db.articles
    docs_needed = list(docs_needed)
    i = 0
    step = 10000
    pbar = tqdm(total=len(docs_needed))
    while i <= len(docs_needed):
        doc_cursor = collection.find({"pmid": {"$in": docs_needed[i:i + step]}})
        for doc in doc_cursor:
            del doc['_id']
            docset[doc['pmid']] = json.loads((json.dumps(doc)))
        i += step
        pbar.update(step)
    pbar.close()
    not_found = set(docs_needed) - set(docset.keys())
    print(list(not_found)[:100])
    print(len(not_found))
    return docset


def create_doc_subset(docset, ret_docs_needed, rel_docs_needed):
    doc_subset = {}
    for doc_id in ret_docs_needed:
        doc_subset[doc_id] = docset[doc_id]
    for doc_id in rel_docs_needed:
        try:
            doc_subset[doc_id] = docset[doc_id]
        except KeyError:
            print('Relevant doc {0} not found in docset.'.format(doc_id))
    return doc_subset


def load_qret(retrieval_results_path):
    f = open(retrieval_results_path, 'r')
    retrieval_results = defaultdict(list)
    for line in f:
        line_splits = line.split()
        q_id = line_splits[0]
        doc_id = line_splits[2]
        bm25_score = float(line_splits[4])
        retrieval_results[q_id].append((doc_id, bm25_score))
    f.close()
    retrieval_results = dict(retrieval_results)
    return retrieval_results


def load_q_text(retrieval_results_path):
    f = open(retrieval_results_path, 'r', errors='ignore')
    q_text = {}
    for line in f:
        line_splits = line.split()
        q_id = line_splits[0]
        text = ' '.join(line_splits[1:])
        q_text[q_id] = text
    f.close()
    q_text = dict(q_text)
    return q_text


def load_q_rels_from_json(retrieval_results_path):
    f = open(retrieval_results_path, 'r')
    data = json.load(f)
    qrels = defaultdict(list)
    n_qrels = defaultdict(int)
    for i in range(len(data['questions'])):
        q_id = data['questions'][i]['id']
        rel_docs = set([doc.split('/')[-1] for doc in data['questions'][i]['documents']])
        qrels[q_id] = rel_docs
        n_qrels[q_id] = len(rel_docs)
    f.close()
    qrels = dict(qrels)
    n_qrels = dict(n_qrels)
    return qrels, n_qrels


def load_q_text_from_json(retrieval_results_path):
    f = open(retrieval_results_path, 'r')
    data = json.load(f)
    q_text = {}
    for i in range(len(data['questions'])):
        q_id = data['questions'][i]['id']
        text = data['questions'][i]['body']
        q_text[q_id] = text
    f.close()
    q_text = dict(q_text)
    return q_text


def add_normalized_scores(q_ret):
    for q in q_ret:
        scores = [t[1] for t in q_ret[q]]
        if np.std(scores) == 0:
            print(q)
        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        if scores_std != 0:
            norm_scores = (scores - scores_mean) / scores_std
        else:
            norm_scores = scores
        for i in range(len(q_ret[q])):
            q_ret[q][i] += (norm_scores[i],)


def remove_recent_years(q_ret, keep_up_to_year):
    new_q_ret = defaultdict(list)
    for q in q_ret:
        for t in q_ret[q]:
            # print(t)
            doc_id = t[0]
            try:
                pub_year = int(docset[doc_id]['publicationDate'].split('-')[0])
            except ValueError:
                continue
            if pub_year > keep_up_to_year:
                print(pub_year)
                continue
            new_q_ret[q].append(t)
    return new_q_ret


def generate_data(queries_file, retrieval_results_path, suffix, keep_up_to_year):
    q_rel, n_qrels = load_q_rels_from_json(queries_file)
    q_ret = load_qret(retrieval_results_path)
    q_text = load_q_text_from_json(queries_file)
    #
    docs_needed = set()
    for q_id in q_rel:
        docs_needed.update(q_rel[q_id])
    for q_id in q_ret:
        docs_needed.update([d[0] for d in q_ret[q_id]])
    docset = create_docset(docs_needed)
    #
    q_ret = remove_recent_years(q_ret, keep_up_to_year)
    #
    for k in [100]:
        print(k)
        #
        for q in q_ret:
            q_ret[q] = q_ret[q][:k]
        add_normalized_scores(q_ret)
        #
        queries = []
        retrieved_documents_set = set()
        relevant_documents_set = set()
        for q in q_ret:
            query_data = {}
            query_data['query_id'] = q
            query_data['query_text'] = q_text[q]
            query_data['relevant_documents'] = sorted(list(q_rel[q]))
            query_data['num_rel'] = n_qrels[q]
            query_data['retrieved_documents'] = []
            rank = 0
            n_ret_rel = 0
            n_ret = 0
            for t in q_ret[q][:k]:
                n_ret += 1
                doc_id = t[0]
                bm25_score = t[1]
                norm_bm25_score = t[2]
                rank += 1
                #
                retrieved_documents_set.add(doc_id)
                relevant_documents_set.update(q_rel[q])
                #
                doc_data = {}
                doc_data['doc_id'] = doc_id
                doc_data['rank'] = rank
                doc_data['bm25_score'] = bm25_score
                doc_data['norm_bm25_score'] = norm_bm25_score
                if doc_id in q_rel[q]:
                    doc_data['is_relevant'] = True
                    n_ret_rel += 1
                else:
                    doc_data['is_relevant'] = False
                query_data['retrieved_documents'].append(doc_data)
            query_data['num_ret'] = n_ret
            query_data['num_rel_ret'] = n_ret_rel
            queries.append(query_data)
            data = {'queries': queries}
        with open('bioasq_bm25_top{0}.{1}.pkl'.format(k, suffix), 'wb') as f:
            pickle.dump(data, f, protocol=2)
        # Create doc subset for the top-k documents (to avoid many queries to mongodb for each k value)
        doc_subset = create_doc_subset(docset, retrieved_documents_set, relevant_documents_set)
        with open('bioasq_bm25_docset_top{0}.{1}.pkl'.format(k, suffix), 'wb') as f:
            pickle.dump(doc_subset, f, protocol=2)


if __name__ == '__main__':
    # generate_data('../bioasq.train.json', 'bioasq_bm25_retrieval.train.txt', 'train', 2015)
    generate_data('../bioasq.dev.json', 'bioasq_bm25_retrieval.dev.txt', 'dev', 2016)
    # generate_data('../bioasq.test.json', 'bioasq_bm25_retrieval.test.txt', 'test', 2016)
