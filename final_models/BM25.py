import json, math
from operator import itemgetter

write_results = True
console_print = True

# Return number of documents
def get_num_of_documents(documents):
    return len(documents)

# Initialize a directory with zeros
def initialize_dictionary(dict):
    for entry, value in dict.items():
        dict[entry] = 0
    return dict

# Return the number of relevants given a list of labels
def num_of_relevant_docs(labels):
    num = 0
    for label in labels:
        if label == 1:
            num += 1
    # To avoid divide with zero
    if num == 0:
        num = 1
    return num

# Compute the AP given a list of returned snippets
def average_Precision(labels):
    avg_precision = 0.0
    count = 1
    retrieved_rel = 0
    for retrieved_doc in labels:
        if retrieved_doc == 1:
            retrieved_rel += 1
            avg_precision += retrieved_rel / count
        count += 1
    avg_precision /= num_of_relevant_docs(labels)
    return avg_precision

# Compute the RR given a list of return snippets
def reciprocal_rank(labels):
    rr = 0
    count = 1
    for label in labels:
        if label == 1:
            rr = 1/count
            return rr
        else:
            count += 1
    return rr

# Compute the term frequency of a word for a specific document
def tf(term, document):
    tf = 0
    for word in document:
        if word == term:
            tf += 1
    if len(document) == 0:
        return tf
    else:
        return tf/len(document)

# Compute the average length from a collection of documents
def compute_avgdl(documents):
    total_words = 0
    for document in documents:
        total_words += len(document)
    avgdl = total_words / len(documents)
    return avgdl

# Compute mean and deviation for Z-score noralization
def compute_Zscore_values(dataset, idf_scores, avgdl, k1, b, rare_word):
    s1s, s2s, labels = [], [], []
    BM25scores = []
    with open(dataset, 'r') as f:
        for line in f:
            items = line[:-1].split('\t')
            s1 = items[1].lower().split()
            s2 = items[2].lower().split()
            label = int(items[3])
            s1s.append(s1)
            s2s.append(s2)
            labels.append(label)
            BM25score = similarity_score(s1, s2, k1, b, idf_scores, avgdl, False, 0, 0, rare_word)
            BM25scores.append(BM25score)
    mean = sum(BM25scores)/ len(BM25scores)
    nominator = 0
    for score in BM25scores:
        nominator += ((score - mean) ** 2)
    deviation = math.sqrt((nominator) / len(BM25scores) - 1)
    return mean, deviation

# Use BM25 ranking function in order to cimpute the similarity score between a question anda snippet
# query: the given question
# document: the snippet
# k1, b: parameters
# idf_scores: list with the idf scores
# avddl: average document length
# nomalize: in case we want to use Z-score normalization (Boolean)
# mean, deviation: variables used for Z-score normalization
def similarity_score(query, document, k1, b, idf_scores, avgdl, normalize, mean, deviation, rare_word):
    score = 0
    for query_term in query:
        if query_term not in idf_scores:
            score += rare_word * ((tf(query_term, document) * (k1 + 1)) / (tf(query_term, document) + k1 * (1 - b + b * (len(document) / avgdl))))
        else:
            score += idf_scores[query_term] * ((tf(query_term, document) * (k1 + 1)) / (tf(query_term, document) + k1 * (1 - b + b * (len(document) / avgdl))))
    if normalize:
        return ((score - mean)/deviation)
    else:
        return score

# Create final file for submission using the BioASQ format
def createBioASQformat(dataset, idf_scores, avgdl, normalized, mean, deviation, rare_word):
    with open(dataset, 'r') as f1:
        scores, s1s, s2s, qids, old_questions, old_answers, starts, ends, docs, bm25_scores = [], [], [], [], [], [], [], [], [], []
        QA_pairs, Bio_pairs = {}, {}
        if (True):
            for line in f1:
                items = line[:-1].split("\t")
                s1 = items[1].lower()
                s2 = items[2].lower().split()
                label = "?"
                qid = items[0]
                old_question = items[3]
                old_answer = items[4]
                start = items[5]
                end = items[6]
                did = items[7]
                s1s.append(s1)
                s2s.append(s2)
                qids.append(qid)
                old_questions.append(old_question)
                old_answers.append(old_answer)
                starts.append(start)
                ends.append(end)
                docs.append(did)
                BM25score = similarity_score(s1, s2, 1.2, 0.75, idf_scores, avgdl, normalized, mean, deviation, rare_word)
                bm25_scores.append(BM25score)
                if s1 in QA_pairs:
                    QA_pairs[s1].append((s2, label, BM25score))
                    Bio_pairs[(s1, qid, old_question)].append((s2, BM25score, old_question, old_answer, start, end, did))
                else:
                    QA_pairs[s1] = [(s2, label, BM25score)]
                    Bio_pairs[(s1, qid, old_question)] = [(s2, BM25score, old_question, old_answer, start, end, did)]
    f2 = open('BM25_predictions.json', 'w')
    data = {'questions': []}
    for keys, candidates in Bio_pairs.items():
        basic_info = {'body': keys[2], 'id': keys[1], 'snippets': []}
        counter = 0
        for answer, pred, old_q, old_a, os, oe, doc_id in sorted(candidates, key = itemgetter(1), reverse = True):
            if counter < 10:
                snips = { 'documents': "http://www.ncbi.nlm.nih.gov/pubmed/" + doc_id,
                          'text': old_a,
                          'offsetInBeginSection': int(os),
                          'offsetInEndSection': int(oe),
                          'beginSection': "abstract",
                          'endSection': "abstract"}
                counter += 1
                basic_info['snippets'].append(snips)
        data['questions'].append(basic_info)
    json.dump(data, f2, indent = 4)


