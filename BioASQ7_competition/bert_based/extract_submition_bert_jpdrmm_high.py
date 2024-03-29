#!/usr/bin/env python
# -*- coding: utf-8 -*-

import  torch, pickle, os, re, nltk, logging, subprocess, json, math, random, time, sys
import  torch.nn.functional     as F
import  numpy                   as np
from    nltk.tokenize           import sent_tokenize
from    tqdm                    import tqdm
from    sklearn.preprocessing   import LabelEncoder
from    sklearn.preprocessing   import OneHotEncoder
from    difflib                 import SequenceMatcher
from    pprint                  import pprint
import  torch.nn                as nn
import  torch.optim             as optim
import  torch.autograd          as autograd
from    pytorch_pretrained_bert.tokenization    import BertTokenizer
from    pytorch_pretrained_bert.modeling        import BertForSequenceClassification
from    pytorch_pretrained_bert.file_utils      import PYTORCH_PRETRAINED_BERT_CACHE

softmax         = lambda z: np.exp(z) / np.sum(np.exp(z))
stopwords       = nltk.corpus.stopwords.words("english")
bioclean        = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def init_the_logger(hdlr):
    if not os.path.exists(odir):
        os.makedirs(odir)
    od = odir.split('/')[-1]  # 'sent_posit_drmm_MarginRankingLoss_0p001'
    logger = logging.getLogger(od)
    if (hdlr is not None):
        logger.removeHandler(hdlr)
    hdlr = logging.FileHandler(os.path.join(odir, 'model.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr

def create_one_hot_and_sim(tokens1, tokens2):
    '''
    :param tokens1:
    :param tokens2:
    :return:
    exxample call : create_one_hot_and_sim('c d e'.split(), 'a b c'.split())
    '''
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    #
    values = list(set(tokens1 + tokens2))
    integer_encoded = label_encoder.fit_transform(values)
    #
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)
    #
    lab1 = label_encoder.transform(tokens1)
    lab1 = np.expand_dims(lab1, axis=1)
    oh1 = onehot_encoder.transform(lab1)
    #
    lab2 = label_encoder.transform(tokens2)
    lab2 = np.expand_dims(lab2, axis=1)
    oh2 = onehot_encoder.transform(lab2)
    #
    ret = np.matmul(oh1, np.transpose(oh2), out=None)
    #
    return oh1, oh2, ret

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    ####
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = bert_tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = bert_tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        ####
        tokens          = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids     = [0] * len(tokens)
        ####
        if tokens_b:
            tokens      += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids       = bert_tokenizer.convert_tokens_to_ids(tokens)
        ####
        input_mask      = [1] * len(input_ids)
        ####
        padding         = [0] * (max_seq_length - len(input_ids))
        input_ids       += padding
        input_mask      += padding
        segment_ids     += padding
        ####
        assert len(input_ids)   == max_seq_length
        assert len(input_mask)  == max_seq_length
        assert len(segment_ids) == max_seq_length
        ####
        in_f        = InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=0)
        in_f.tokens = tokens
        features.append(in_f)
    return features

def tf(term, document):
    tf = 0
    for word in document:
        if word == term:
            tf += 1
    if len(document) == 0:
        return tf
    else:
        return tf / len(document)

def compute_the_cost(optimizer, costs, back_prop=True):
    cost_ = torch.stack(costs)
    cost_ = cost_.sum() / (1.0 * cost_.size(0))
    if (back_prop):
        cost_.backward()
        optimizer.step()
        optimizer.zero_grad()
    the_cost = cost_.cpu().item()
    return the_cost

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

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))

def RemoveTrainLargeYears(data, doc_text):
    for i in range(len(data['queries'])):
        hyear = 1900
        for j in range(len(data['queries'][i]['retrieved_documents'])):
            if data['queries'][i]['retrieved_documents'][j]['is_relevant']:
                doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
                year = doc_text[doc_id]['publicationDate'].split('-')[0]
                if year[:1] == '1' or year[:1] == '2':
                    if int(year) > hyear:
                        hyear = int(year)
        j = 0
        while True:
            doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
            year = doc_text[doc_id]['publicationDate'].split('-')[0]
            if (year[:1] == '1' or year[:1] == '2') and int(year) > hyear:
                del data['queries'][i]['retrieved_documents'][j]
            else:
                j += 1
            if j == len(data['queries'][i]['retrieved_documents']):
                break
    return data

def RemoveBadYears(data, doc_text, train):
    for i in range(len(data['queries'])):
        j = 0
        while True:
            doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
            year = doc_text[doc_id]['publicationDate'].split('-')[0]
            ##########################
            # Skip 2017/2018 docs always. Skip 2016 docs for training.
            # Need to change for final model - 2017 should be a train year only.
            # Use only for testing.
            if year == '2017' or year == '2018' or (train and year == '2016'):
                # if year == '2018' or (train and year == '2017'):
                del data['queries'][i]['retrieved_documents'][j]
            else:
                j += 1
            ##########################
            if j == len(data['queries'][i]['retrieved_documents']):
                break
    return data

def get_words(s):
    sl = tokenize(s)
    sl = [s for s in sl]
    sl2 = [s for s in sl if idf_val(s) >= 2.0]
    return sl, sl2

def fix_bert_tokens(tokens):
    ret = []
    for t in tokens:
        if (t.startswith('##')):
            ret[-1] = ret[-1] + t[2:]
        else:
            ret.append(t)
    return ret

def tokenize(x):
    x_tokens = bert_tokenizer.tokenize(x)
    x_tokens = fix_bert_tokens(x_tokens)
    return x_tokens

def idf_val(w):
    if w in idf:
        return idf[w]
    return max_idf

def get_embeds(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if (tok in wv):
            ret1.append(tok)
            ret2.append(wv[tok])
    return ret1, np.array(ret2, 'float64')

def load_idfs(idf_path, words):
    print('Loading IDF tables')
    #
    # with open(dataloc + 'idf.pkl', 'rb') as f:
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    ret = {}
    for w in words:
        if w in idf:
            ret[w] = idf[w]
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    idf = None
    print('Loaded idf tables with max idf {}'.format(max_idf))
    #
    return ret, max_idf

def uwords(words):
    uw = {}
    for w in words:
        uw[w] = 1
    return [w for w in uw]

def ubigrams(words):
    uw = {}
    prevw = "<pw>"
    for w in words:
        uw[prevw + '_' + w] = 1
        prevw = w
    return [w for w in uw]

def query_doc_overlap(qwords, dwords):
    # % Query words in doc.
    qwords_in_doc = 0
    idf_qwords_in_doc = 0.0
    idf_qwords = 0.0
    for qword in uwords(qwords):
        idf_qwords += idf_val(qword)
        for dword in uwords(dwords):
            if qword == dword:
                idf_qwords_in_doc += idf_val(qword)
                qwords_in_doc += 1
                break
    if len(qwords) <= 0:
        qwords_in_doc_val = 0.0
    else:
        qwords_in_doc_val = (float(qwords_in_doc) /
                             float(len(uwords(qwords))))
    if idf_qwords <= 0.0:
        idf_qwords_in_doc_val = 0.0
    else:
        idf_qwords_in_doc_val = float(idf_qwords_in_doc) / float(idf_qwords)
    # % Query bigrams  in doc.
    qwords_bigrams_in_doc = 0
    idf_qwords_bigrams_in_doc = 0.0
    idf_bigrams = 0.0
    for qword in ubigrams(qwords):
        wrds = qword.split('_')
        idf_bigrams += idf_val(wrds[0]) * idf_val(wrds[1])
        for dword in ubigrams(dwords):
            if qword == dword:
                qwords_bigrams_in_doc += 1
                idf_qwords_bigrams_in_doc += (idf_val(wrds[0]) * idf_val(wrds[1]))
                break
    if len(qwords) <= 0:
        qwords_bigrams_in_doc_val = 0.0
    else:
        qwords_bigrams_in_doc_val = (float(qwords_bigrams_in_doc) / float(len(ubigrams(qwords))))
    if idf_bigrams <= 0.0:
        idf_qwords_bigrams_in_doc_val = 0.0
    else:
        idf_qwords_bigrams_in_doc_val = (float(idf_qwords_bigrams_in_doc) / float(idf_bigrams))
    return [
        qwords_in_doc_val,
        qwords_bigrams_in_doc_val,
        idf_qwords_in_doc_val,
        idf_qwords_bigrams_in_doc_val
    ]

def GetScores(qtext, dtext, bm25):
    qwords, qw2 = get_words(qtext)
    dwords, dw2 = get_words(dtext)
    qd1 = query_doc_overlap(qwords, dwords)
    bm25 = [bm25]
    return qd1[0:3] + bm25

def GetWords(data, doc_text, words):
    for i in tqdm(range(len(data['queries'])), ascii=True):
        qwds = tokenize(data['queries'][i]['query_text'])
        for w in qwds:
            words[w] = 1
        for j in range(len(data['queries'][i]['retrieved_documents'])):
            doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
            dtext = (doc_text[doc_id]['title'] + ' <title> ' + doc_text[doc_id]['abstractText'])
            dwds = tokenize(dtext)
            for w in dwds:
                words[w] = 1

def prep_extracted_snippets(extracted_snippets, docs, qid, top10docs, quest_body):
    ret = {
        'body': quest_body,
        'documents': top10docs,
        'id': qid,
        'snippets': [],
    }
    for esnip in extracted_snippets:
        pid = esnip[2].split('/')[-1]
        the_text = esnip[3]
        esnip_res = {
            # 'score'     : esnip[1],
            "document": "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pid),
            "text": the_text
        }
        try:
            ind_from = docs[pid]['title'].index(the_text)
            ind_to = ind_from + len(the_text)
            esnip_res["beginSection"] = "title"
            esnip_res["endSection"] = "title"
            esnip_res["offsetInBeginSection"] = ind_from
            esnip_res["offsetInEndSection"] = ind_to
        except:
            # print(the_text)
            # pprint(docs[pid])
            ind_from = docs[pid]['abstractText'].index(the_text)
            ind_to = ind_from + len(the_text)
            esnip_res["beginSection"] = "abstract"
            esnip_res["endSection"] = "abstract"
            esnip_res["offsetInBeginSection"] = ind_from
            esnip_res["offsetInEndSection"] = ind_to
        ret['snippets'].append(esnip_res)
    return ret

def get_snips(quest_id, gid, bioasq6_data):
    good_snips = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            if (sn['document'].endswith(gid)):
                good_snips.extend(sent_tokenize(sn['text']))
    return good_snips

def snip_is_relevant(one_sent, gold_snips):
    return int(
        any(
            [
                (one_sent.encode('ascii', 'ignore') in gold_snip.encode('ascii', 'ignore'))
                or
                (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii', 'ignore'))
                for gold_snip in gold_snips
            ]
        )
    )

def get_norm_doc_scores(the_doc_scores):
    ks = list(the_doc_scores.keys())
    vs = [the_doc_scores[k] for k in ks]
    vs = softmax(vs)
    norm_doc_scores = {}
    for i in range(len(ks)):
        norm_doc_scores[ks[i]] = vs[i]
    return norm_doc_scores

def select_snippets_v3(extracted_snippets, the_doc_scores):
    '''
    :param      extracted_snippets:
    :param      doc_res:
    :return:    returns the top 10 snippets across all documents (0..n from each doc)
    '''
    norm_doc_scores = get_norm_doc_scores(the_doc_scores)
    # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
    extracted_snippets = [tt for tt in extracted_snippets if (tt[2] in norm_doc_scores)]
    sorted_snips = sorted(extracted_snippets, key=lambda x: x[1] * norm_doc_scores[x[2]], reverse=True)
    return sorted_snips[:10]

def similar(upstream_seq, downstream_seq):
    upstream_seq = upstream_seq.encode('ascii', 'ignore')
    downstream_seq = downstream_seq.encode('ascii', 'ignore')
    s = SequenceMatcher(None, upstream_seq, downstream_seq)
    match = s.find_longest_match(0, len(upstream_seq), 0, len(downstream_seq))
    upstream_start = match[0]
    upstream_end = match[0] + match[2]
    longest_match = upstream_seq[upstream_start:upstream_end]
    to_match = upstream_seq if (len(downstream_seq) > len(upstream_seq)) else downstream_seq
    r1 = SequenceMatcher(None, to_match, longest_match).ratio()
    return r1

def save_checkpoint(epoch, model, bert_model, max_dev_map, optimizer1, optimizer2, filename='checkpoint.pth.tar'):
    state = {
        'epoch'             : epoch,
        'model_state_dict'  : model.state_dict(),
        'bert_state_dict'   : bert_model.state_dict(),
        'best_valid_score'  : max_dev_map,
        'optimizer1'        : optimizer1.state_dict(),
        'optimizer2'        : optimizer2.state_dict() if(optimizer2 is not None) else None,
    }
    torch.save(state, filename)

def embed_the_sent(sent):
    eval_examples   = [InputExample(guid='example_dato_1', text_a=sent, text_b=None, label='1')]
    eval_features   = convert_examples_to_features(eval_examples)
    eval_feat       = eval_features[0]
    input_ids       = torch.tensor([eval_feat.input_ids], dtype=torch.long).to(device)
    input_mask      = torch.tensor([eval_feat.input_mask], dtype=torch.long).to(device)
    segment_ids     = torch.tensor([eval_feat.segment_ids], dtype=torch.long).to(device)
    tokens          = eval_feat.tokens
    with torch.no_grad():
        token_embeds, pooled_output = bert_model.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        tok_inds                    = [i for i in range(len(tokens)) if(not tokens[i].startswith('##'))]
        token_embeds                = token_embeds.squeeze(0)
        embs                        = token_embeds[tok_inds,:]
    fixed_tokens = fix_bert_tokens(tokens)
    return fixed_tokens, embs

def get_map_res(fgold, femit, eval_path):
    trec_eval_res = subprocess.Popen(['python', eval_path, fgold, femit], stdout=subprocess.PIPE, shell=False)
    (out, err) = trec_eval_res.communicate()
    lines = out.decode("utf-8").split('\n')
    map_res = [l for l in lines if (l.startswith('map '))][0].split('\t')
    map_res = float(map_res[-1])
    return map_res

def get_bioasq_res(prefix, data_gold, data_emitted, data_for_revision):
    '''
    java -Xmx10G -cp /home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar
    evaluation.EvaluatorTask1b -phaseA -e 5
    /home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/BioASQ-task6bPhaseB-testset1
    ./drmm-experimental_submit.json
    '''
    jar_path = retrieval_jar_path
    #
    fgold = '{}_data_for_revision.json'.format(prefix)
    fgold = os.path.join(odir, fgold)
    fgold = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_for_revision, indent=4, sort_keys=True))
        f.close()
    #
    for tt in data_gold['questions']:
        if ('exact_answer' in tt):
            del (tt['exact_answer'])
        if ('ideal_answer' in tt):
            del (tt['ideal_answer'])
        if ('type' in tt):
            del (tt['type'])
    fgold = '{}_gold_bioasq.json'.format(prefix)
    fgold = os.path.join(odir, fgold)
    fgold = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_gold, indent=4, sort_keys=True))
        f.close()
    #
    femit = '{}_emit_bioasq.json'.format(prefix)
    femit = os.path.join(odir, femit)
    femit = os.path.abspath(femit)
    with open(femit, 'w') as f:
        f.write(json.dumps(data_emitted, indent=4, sort_keys=True))
        f.close()
    #
    bioasq_eval_res = subprocess.Popen(
        [
            'java', '-Xmx10G', '-cp', jar_path, 'evaluation.EvaluatorTask1b',
            '-phaseA', '-e', '5', fgold, femit
        ],
        stdout=subprocess.PIPE, shell=False
    )
    (out, err) = bioasq_eval_res.communicate()
    lines = out.decode("utf-8").split('\n')
    ret = {}
    for line in lines:
        if (':' in line):
            k = line.split(':')[0].strip()
            v = line.split(':')[1].strip()
            ret[k] = float(v)
    return ret

def load_all_data(dataloc, idf_pickle_path, bert_all_words_path):
    print('loading pickle data')
    #
    with open(dataloc + 'trainining7b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict((q['id'], q) for q in bioasq6_data['questions'])
    #
    with open(dataloc + 'bioasq7_bm25_top100.dev.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_docset_top100.dev.pkl', 'rb') as f:
        dev_docs = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_top100.train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataloc + 'bioasq7_bm25_docset_top100.train.pkl', 'rb') as f:
        train_docs = pickle.load(f)
    print('loading words')
    #
    if (os.path.exists(bert_all_words_path)):
        words = pickle.load(open(bert_all_words_path, 'rb'))
    else:
        words = {}
        GetWords(train_data, train_docs, words)
        GetWords(dev_data, dev_docs, words)
        pickle.dump(words, open(bert_all_words_path, 'wb'), protocol=2)
    #
    print('loading idfs')
    idf, max_idf = load_idfs(idf_pickle_path, words)
    return dev_data, dev_docs, train_data, train_docs, idf, max_idf, bioasq6_data

def do_for_one_retrieved(doc_emit_, gs_emits_, held_out_sents, retr, doc_res, gold_snips):
    emition = doc_emit_.cpu().item()
    emitss = gs_emits_.tolist()
    mmax = max(emitss)
    all_emits, extracted_from_one = [], []
    for ind in range(len(emitss)):
        t = (
            snip_is_relevant(held_out_sents[ind], gold_snips),
            emitss[ind],
            "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(retr['doc_id']),
            held_out_sents[ind]
        )
        all_emits.append(t)
        # if (emitss[ind] == mmax):
        #     extracted_from_one.append(t)
        extracted_from_one.append(t)
    doc_res[retr['doc_id']] = float(emition)
    all_emits = sorted(all_emits, key=lambda x: x[1], reverse=True)
    return doc_res, extracted_from_one, all_emits

def prep_data(quest, the_doc, the_bm25, good_snips, quest_toks):
    good_sents          = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
    ####
    good_doc_af         = GetScores(quest, the_doc['title'] + the_doc['abstractText'], the_bm25)
    good_doc_af.append(len(good_sents) / 60.)
    #
    all_doc_text        = the_doc['title'] + ' ' + the_doc['abstractText']
    doc_toks            = tokenize(all_doc_text)
    tomi                = (set(doc_toks) & set(quest_toks))
    tomi_no_stop        = tomi - set(stopwords)
    BM25score           = similarity_score(quest_toks, doc_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
    tomi_no_stop_idfs   = [idf_val(w) for w in tomi_no_stop]
    tomi_idfs           = [idf_val(w) for w in tomi]
    quest_idfs          = [idf_val(w) for w in quest_toks]
    features            = [
        len(quest) / 300.,
        len(all_doc_text) / 300.,
        len(tomi_no_stop) / 100.,
        BM25score,
        sum(tomi_no_stop_idfs) / 100.,
        sum(tomi_idfs) / sum(quest_idfs),
    ]
    good_doc_af.extend(features)
    ####
    good_sents_embeds, good_sents_escores, held_out_sents, good_sent_tags, good_oh_sim = [], [], [], [], []
    for good_text in good_sents:
        sent_toks, sent_embeds  = embed_the_sent(' '.join(bioclean(good_text)))
        oh1, oh2, oh_sim        = create_one_hot_and_sim(quest_toks, sent_toks)
        good_oh_sim.append(oh_sim)
        good_escores            = GetScores(quest, good_text, the_bm25)[:-1]
        good_escores.append(len(sent_toks) / 342.)
        tomi                    = (set(sent_toks) & set(quest_toks))
        tomi_no_stop            = tomi - set(stopwords)
        BM25score               = similarity_score(quest_toks, sent_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
        tomi_no_stop_idfs       = [idf_val(w) for w in tomi_no_stop]
        tomi_idfs               = [idf_val(w) for w in tomi]
        quest_idfs              = [idf_val(w) for w in quest_toks]
        features                = [
            len(quest) / 300.,
            len(good_text) / 300.,
            len(tomi_no_stop) / 100.,
            BM25score,
            sum(tomi_no_stop_idfs) / 100.,
            sum(tomi_idfs) / sum(quest_idfs),
        ]
        #
        good_sents_embeds.append(sent_embeds)
        good_sents_escores.append(good_escores + features)
        held_out_sents.append(good_text)
        good_sent_tags.append(snip_is_relevant(' '.join(bioclean(good_text)), good_snips))
    ####
    return {
        'sents_embeds': good_sents_embeds,
        'sents_escores': good_sents_escores,
        'doc_af': good_doc_af,
        'sent_tags': good_sent_tags,
        'held_out_sents': held_out_sents,
        'oh_sims': good_oh_sim
    }

def do_for_some_retrieved(docs, dato, retr_docs, data_for_revision, ret_data, use_sent_tokenizer):
    emitions = {
        'body': dato['query_text'],
        'id': dato['query_id'],
        'documents': []
    }
    ####
    quest_text          = dato['query_text']
    quest_text          = ' '.join(bioclean(quest_text.replace('\ufeff', ' ')))
    quest_tokens, qemb  = embed_the_sent(quest_text)
    ####
    q_idfs              = np.array([[idf_val(qw)] for qw in quest_tokens], 'float')
    gold_snips          = []
    #
    doc_res, extracted_snippets         = {}, []
    extracted_snippets_known_rel_num    = []
    for retr in retr_docs:
        datum                   = prep_data(quest_text, docs[retr['doc_id']], retr['norm_bm25_score'], gold_snips, quest_tokens)
        doc_emit_, gs_emits_ = model.emit_one(
            doc1_sents_embeds   = datum['sents_embeds'],
            doc1_oh_sim         = datum['oh_sims'],
            question_embeds     = qemb,
            q_idfs              = q_idfs,
            sents_gaf           = datum['sents_escores'],
            doc_gaf             = datum['doc_af']
        )
        doc_res, extracted_from_one, all_emits = do_for_one_retrieved(
            doc_emit_, gs_emits_, datum['held_out_sents'], retr, doc_res, gold_snips
        )
        # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
        extracted_snippets.extend(extracted_from_one)
        #
        total_relevant = sum([1 for em in all_emits if (em[0] == True)])
        if (total_relevant > 0):
            extracted_snippets_known_rel_num.extend(all_emits[:total_relevant])
        if (dato['query_id'] not in data_for_revision):
            data_for_revision[dato['query_id']] = {'query_text': dato['query_text'],
                                                   'snippets': {retr['doc_id']: all_emits}}
        else:
            data_for_revision[dato['query_id']]['snippets'][retr['doc_id']] = all_emits
    #
    doc_res = sorted(doc_res.items(), key=lambda x: x[1], reverse=True)
    the_doc_scores = dict([("http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]), pm[1]) for pm in doc_res[:10]])
    doc_res = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
    emitions['documents'] = doc_res[:100]
    ret_data['questions'].append(emitions)
    #
    extracted_snippets = [tt for tt in extracted_snippets if (tt[2] in doc_res[:10])]
    extracted_snippets_known_rel_num = [tt for tt in extracted_snippets_known_rel_num if (tt[2] in doc_res[:10])]
    if (use_sent_tokenizer):
        extracted_snippets_v3 = select_snippets_v3(extracted_snippets, the_doc_scores)
        extracted_snippets_known_rel_num_v3 = select_snippets_v3(extracted_snippets_known_rel_num, the_doc_scores)
    else:
        extracted_snippets_v3 = []
        extracted_snippets_known_rel_num_v3 = []
    #
    snips_res_v3 = prep_extracted_snippets(extracted_snippets_v3, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    #
    snips_res_known_rel_num_v3 = prep_extracted_snippets(extracted_snippets_known_rel_num_v3, docs, dato['query_id'],
                                                         doc_res[:10], dato['query_text'])
    #
    snips_res = {'v3': snips_res_v3}
    snips_res_known = {'v3': snips_res_known_rel_num_v3}
    return data_for_revision, ret_data, snips_res, snips_res_known

def print_the_results(prefix, all_bioasq_gold_data, all_bioasq_subm_data, all_bioasq_subm_data_known, data_for_revision):
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data_known, data_for_revision)
    pprint(bioasq_snip_res)
    print('{} known MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    print('{} known F1 snippets: {}'.format(prefix, bioasq_snip_res['MF1 snippets']))
    print('{} known MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    print('{} known GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data, data_for_revision)
    pprint(bioasq_snip_res)
    print('{} MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    print('{} F1 snippets: {}'.format(prefix, bioasq_snip_res['MF1 snippets']))
    print('{} MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    print('{} GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #

def print_params(model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(model)
    print(40 * '=')
    trainable       = 0
    untrainable     = 0
    for parameter in model.parameters():
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        if(parameter.requires_grad):
            trainable   += v
        else:
            untrainable += v
    total_params = trainable + untrainable
    print(40 * '=')
    print('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    print(40 * '=')

def get_one_map(prefix, data, docs, use_sent_tokenizer):
    model.eval()
    bert_model.eval()
    #
    ret_data                        = {'questions': []}
    all_bioasq_subm_data_v3         = {"questions": []}
    all_bioasq_subm_data_known_v3   = {"questions": []}
    all_bioasq_gold_data            = {'questions': []}
    data_for_revision               = {}
    #
    for dato in tqdm(data['queries'], ascii=True):
        all_bioasq_gold_data['questions'].append(bioasq7_data[dato['query_id']])
        data_for_revision, ret_data, snips_res, snips_res_known = do_for_some_retrieved(
            docs, dato, dato['retrieved_documents'], data_for_revision, ret_data,use_sent_tokenizer)
        all_bioasq_subm_data_v3['questions'].append(snips_res['v3'])
        all_bioasq_subm_data_known_v3['questions'].append(snips_res_known['v3'])
    #
    print_the_results('v3 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v3, all_bioasq_subm_data_known_v3,
                      data_for_revision)
    #

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim=30, k_for_maxpool=5, sentence_out_method='MLP', k_sent_maxpool=1):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k = k_for_maxpool
        self.k_sent_maxpool = k_sent_maxpool
        self.doc_add_feats = 11
        self.sent_add_feats = 10
        #
        self.embedding_dim = embedding_dim
        self.sentence_out_method = sentence_out_method
        # to create q weights
        self.init_context_module()
        self.init_question_weight_module()
        self.init_mlps_for_pooled_attention()
        self.init_sent_output_layer()
        self.init_doc_out_layer()
        # doc loss func
        self.margin_loss = nn.MarginRankingLoss(margin=1.0).to(device)

    def init_mesh_module(self):
        self.mesh_h0 = autograd.Variable(torch.randn(1, 1, self.embedding_dim)).to(device)
        self.mesh_gru = nn.GRU(self.embedding_dim, self.embedding_dim).to(device)

    def init_context_module(self):
        self.trigram_conv_1 = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True).to(device)
        self.trigram_conv_activation_1 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
        self.trigram_conv_2 = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True).to(device)
        self.trigram_conv_activation_2 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)

    def init_question_weight_module(self):
        self.q_weights_mlp = nn.Linear(self.embedding_dim + 1, 1, bias=True).to(device)

    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1 = nn.Linear(3 * 3, 8, bias=True).to(device)
        self.my_relu1 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
        self.linear_per_q2 = nn.Linear(8, 1, bias=True).to(device)

    def init_sent_output_layer(self):
        if (self.sentence_out_method == 'MLP'):
            self.sent_out_layer_1 = nn.Linear(self.sent_add_feats + 1, 8, bias=False).to(device)
            self.sent_out_activ_1 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
            self.sent_out_layer_2 = nn.Linear(8, 1, bias=False).to(device)
        else:
            self.sent_res_h0 = autograd.Variable(torch.randn(2, 1, 5)).to(device)
            self.sent_res_bigru = nn.GRU(input_size=self.sent_add_feats + 1, hidden_size=5, bidirectional=True,
                                         batch_first=False).to(device)
            self.sent_res_mlp = nn.Linear(10, 1, bias=False).to(device)

    def init_doc_out_layer(self):
        self.final_layer_1 = nn.Linear(self.doc_add_feats + self.k_sent_maxpool, 8, bias=True).to(device)
        self.final_activ_1 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
        self.final_layer_2 = nn.Linear(8, 1, bias=True).to(device)
        self.oo_layer = nn.Linear(2, 1, bias=True).to(device)

    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos

    def apply_context_gru(self, the_input, h0):
        output, hn = self.context_gru(the_input.unsqueeze(1), h0)
        output = self.context_gru_activation(output)
        out_forward = output[:, 0, :self.embedding_dim]
        out_backward = output[:, 0, self.embedding_dim:]
        output = out_forward + out_backward
        res = output + the_input
        return res, hn

    def apply_context_convolution(self, the_input, the_filters, activation):
        conv_res = the_filters(the_input.transpose(0, 1).unsqueeze(0))
        if (activation is not None):
            conv_res = activation(conv_res)
        pad = the_filters.padding[0]
        ind_from = int(np.floor(pad / 2.0))
        ind_to = ind_from + the_input.size(0)
        conv_res = conv_res[:, :, ind_from:ind_to]
        conv_res = conv_res.transpose(1, 2)
        conv_res = conv_res + the_input
        return conv_res.squeeze(0)

    def my_cosine_sim(self, A, B):
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        A_mag = torch.norm(A, 2, dim=2)
        B_mag = torch.norm(B, 2, dim=2)
        num = torch.bmm(A, B.transpose(-1, -2))
        den = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
        dist_mat = num / den
        return dist_mat

    def pooling_method(self, sim_matrix):
        sorted_res = torch.sort(sim_matrix, -1)[0]  # sort the input minimum to maximum
        k_max_pooled = sorted_res[:, -self.k:]  # select the last k of each instance in our data
        average_k_max_pooled = k_max_pooled.sum(-1) / float(self.k)  # average these k values
        the_maximum = k_max_pooled[:, -1]  # select the maximum value of each instance
        the_average_over_all = sorted_res.sum(-1) / float(sim_matrix.size(1))  # add average of all elements as long sentences might have more matches
        the_concatenation = torch.stack([the_maximum, average_k_max_pooled, the_average_over_all],dim=-1)  # concatenate maximum value and average of k-max values
        return the_concatenation  # return the concatenation

    def get_output(self, input_list, weights):
        temp = torch.cat(input_list, -1)
        lo = self.linear_per_q1(temp)
        lo = self.my_relu1(lo)
        lo = self.linear_per_q2(lo)
        lo = lo.squeeze(-1)
        lo = lo * weights
        sr = lo.sum(-1) / lo.size(-1)
        return sr

    def apply_sent_res_bigru(self, the_input):
        output, hn = self.sent_res_bigru(the_input.unsqueeze(1), self.sent_res_h0)
        output = self.sent_res_mlp(output)
        return output.squeeze(-1).squeeze(-1)

    def do_for_one_doc_cnn(self, doc_sents_embeds, oh_sims, sents_af, question_embeds, q_conv_res_trigram, q_weights, k2):
        res = []
        for i in range(len(doc_sents_embeds)):
            sim_oh = autograd.Variable(torch.FloatTensor(oh_sims[i]), requires_grad=False).to(device)
            sent_embeds = doc_sents_embeds[i]
            gaf = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False).to(device)
            #
            conv_res            = self.apply_context_convolution(sent_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
            conv_res            = self.apply_context_convolution(conv_res, self.trigram_conv_2, self.trigram_conv_activation_2)
            #
            sim_insens          = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_sens            = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled  = self.pooling_method(sim_insens)
            sensitive_pooled    = self.pooling_method(sim_sens)
            oh_pooled           = self.pooling_method(sim_oh)
            #
            sent_emit           = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats      = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        if (self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer_1(res)
            res = self.sent_out_activ_1(res)
            res = self.sent_out_layer_2(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        # ret = self.get_max(res).unsqueeze(0)
        ret = self.get_kmax(res, k2)
        return ret, res

    def do_for_one_doc_bigru(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights, k2):
        res = []
        hn = self.context_h0
        for i in range(len(doc_sents_embeds)):
            sent_embeds = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False).to(device)
            gaf = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False).to(device)
            conv_res, hn = self.apply_context_gru(sent_embeds, hn)
            #
            sim_insens = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_oh = (sim_insens > (1 - (1e-3))).float()
            sim_sens = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled = self.pooling_method(sim_insens)
            sensitive_pooled = self.pooling_method(sim_sens)
            oh_pooled = self.pooling_method(sim_oh)
            #
            sent_emit = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        if (self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer_1(res)
            res = self.sent_out_activ_1(res)
            res = self.sent_out_layer_2(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        # ret = self.get_max(res).unsqueeze(0)
        ret = self.get_kmax(res, k2)
        res = torch.sigmoid(res)
        return ret, res

    def get_max(self, res):
        return torch.max(res)

    def get_kmax(self, res, k):
        res = torch.sort(res, 0)[0]
        res = res[-k:].squeeze(-1)
        if (len(res.size()) == 0):
            res = res.unsqueeze(0)
        if (res.size()[0] < k):
            to_concat = torch.zeros(k - res.size()[0]).to(device)
            res = torch.cat([res, to_concat], -1)
        return res

    def get_max_and_average_of_k_max(self, res, k):
        k_max_pooled = self.get_kmax(res, k)
        average_k_max_pooled = k_max_pooled.sum() / float(k)
        the_maximum = k_max_pooled[-1]
        the_concatenation = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation

    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res

    def get_maxmin_max(self, res):
        res = self.min_max_norm(res)
        res = torch.max(res)
        return res

    def apply_mesh_gru(self, mesh_embeds):
        mesh_embeds = autograd.Variable(torch.FloatTensor(mesh_embeds), requires_grad=False).to(device)
        output, hn = self.mesh_gru(mesh_embeds.unsqueeze(1), self.mesh_h0)
        return output[-1, 0, :]

    def get_mesh_rep(self, meshes_embeds, q_context):
        meshes_embeds = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds = torch.stack(meshes_embeds)
        sim_matrix = self.my_cosine_sim(meshes_embeds, q_context).squeeze(0)
        max_sim = torch.sort(sim_matrix, -1)[0][:, -1]
        output = torch.mm(max_sim.unsqueeze(0), meshes_embeds)[0]
        return output

    def emit_one(self, doc1_sents_embeds, doc1_oh_sim, question_embeds, q_idfs, sents_gaf, doc_gaf):
        q_idfs          = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False).to(device)
        doc_gaf         = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False).to(device)
        #
        q_context = self.apply_context_convolution(question_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context = self.apply_context_convolution(q_context, self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights = torch.cat([q_context, q_idfs], -1)
        q_weights = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits = self.do_for_one_doc_cnn(
            doc1_sents_embeds,
            doc1_oh_sim,
            sents_gaf,
            question_embeds,
            q_context,
            q_weights,
            self.k_sent_maxpool
        )
        #
        good_out_pp = torch.cat([good_out, doc_gaf], -1)
        #
        final_good_output = self.final_layer_1(good_out_pp)
        final_good_output = self.final_activ_1(final_good_output)
        final_good_output = self.final_layer_2(final_good_output)
        #
        gs_emits = gs_emits.unsqueeze(-1)
        gs_emits = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits = torch.sigmoid(gs_emits)
        #
        return final_good_output, gs_emits

    def forward(self, doc1_sents_embeds, doc2_sents_embeds, doc1_oh_sim, doc2_oh_sim,
                question_embeds, q_idfs, sents_gaf, sents_baf, doc_gaf, doc_baf):
        q_idfs = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False).to(device)
        doc_gaf = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False).to(device)
        doc_baf = autograd.Variable(torch.FloatTensor(doc_baf), requires_grad=False).to(device)
        #
        q_context = self.apply_context_convolution(question_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context = self.apply_context_convolution(q_context, self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights = torch.cat([q_context, q_idfs], -1)
        q_weights = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits = self.do_for_one_doc_cnn(
            doc1_sents_embeds,
            doc1_oh_sim,
            sents_gaf,
            question_embeds,
            q_context,
            q_weights,
            self.k_sent_maxpool
        )
        bad_out, bs_emits = self.do_for_one_doc_cnn(
            doc2_sents_embeds,
            doc2_oh_sim,
            sents_baf,
            question_embeds,
            q_context,
            q_weights,
            self.k_sent_maxpool
        )
        #
        good_out_pp = torch.cat([good_out, doc_gaf], -1)
        bad_out_pp = torch.cat([bad_out, doc_baf], -1)
        #
        final_good_output = self.final_layer_1(good_out_pp)
        final_good_output = self.final_activ_1(final_good_output)
        final_good_output = self.final_layer_2(final_good_output)
        #
        gs_emits = gs_emits.unsqueeze(-1)
        gs_emits = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits = torch.sigmoid(gs_emits)
        #
        final_bad_output = self.final_layer_1(bad_out_pp)
        final_bad_output = self.final_activ_1(final_bad_output)
        final_bad_output = self.final_layer_2(final_bad_output)
        #
        bs_emits = bs_emits.unsqueeze(-1)
        # bs_emits = torch.cat([bs_emits, final_good_output.unsqueeze(-1).expand_as(bs_emits)], -1)
        bs_emits = torch.cat([bs_emits, final_bad_output.unsqueeze(-1).expand_as(bs_emits)], -1)
        bs_emits = self.oo_layer(bs_emits).squeeze(-1)
        bs_emits = torch.sigmoid(bs_emits)
        #
        loss1 = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output, gs_emits, bs_emits

def load_model_from_checkpoint(resume_dir):
    global start_epoch, optimizer
    resume_from = os.path.join(resume_dir, 'best_checkpoint.pth.tar')
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        #############################################################################################
        model.load_state_dict(checkpoint['model_state_dict'])
        bert_model.load_state_dict(checkpoint['bert_state_dict'])
        #############################################################################################
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

min_doc_score               = -1000.
min_sent_score              = -1000.
emit_only_abstract_sents    = False
###########################################################
use_cuda            = torch.cuda.is_available()
###########################################################
batch_no            = sys.argv[1]
f_in1               = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseA-testset{}'.format(batch_no, batch_no)
f_in2               = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'.format(batch_no)
f_in3               = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl'.format(batch_no)
odir                = './test_bert_jpdrmm_unfrozen_high_batch{}/'.format(batch_no)
###########################################################
eval_path           = '/home/dpappas/bioasq_all/eval/run_eval.py'
retrieval_jar_path  = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'
odd                 = '/home/dpappas/'
###########################################################
w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
###########################################################
if (not os.path.exists(odir)):
    os.makedirs(odir)
###########################################################
avgdl               = 21.1907
mean                = 0.6275
deviation           = 1.2210
print(avgdl, mean, deviation)
###########################################################
k_for_maxpool       = 5
k_sent_maxpool      = 5
embedding_dim       = 768 #200
###########################################################
my_seed     = 1
random.seed(my_seed)
torch.manual_seed(my_seed)
###########################################################
print('Compiling model...')
max_seq_length      = 50
device              = torch.device("cuda") if(use_cuda) else torch.device("cpu")
bert_model          = 'bert-base-uncased'
cache_dir           = '/home/dpappas/bert_cache/'
bert_tokenizer      = BertTokenizer.from_pretrained(bert_model, do_lower_case=True, cache_dir=cache_dir)
bert_model          = BertForSequenceClassification.from_pretrained(bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1), num_labels=2)
model               = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool)
###########################################################
resume_from         = '/media/dpappas/dpappas_data/models_out/bioasq7_bert_jpdrmm_2L_0p01_unfrozen_run_0/'
load_model_from_checkpoint(resume_from)
for param in model.parameters():
    param.requires_grad = False
for param in bert_model.parameters():
    param.requires_grad = False
print_params(model)
print_params(bert_model)
bert_model.to(device)
model.to(device)
###########################################################
print('loading pickle data')
with open(f_in1, 'r') as f:
    bioasq7_data = json.load(f)
    for q in bioasq7_data['questions']:
        if("documents" not in q):
            q["documents"]  = []
        if("snippets" not in q):
            q["snippets"]   = []
    bioasq7_data = dict((q['id'], q) for q in bioasq7_data['questions'])
with open(f_in2, 'rb') as f:
    test_data = pickle.load(f)
with open(f_in3, 'rb') as f:
    test_docs = pickle.load(f)
###########################################################
words = {}
GetWords(test_data, test_docs, words)
print('loading idfs')
idf, max_idf = load_idfs(idf_pickle_path, words)
###########################################################
test_map        = get_one_map('test', test_data, test_docs, use_sent_tokenizer=True)
print(test_map)


'''
java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' \
evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/media/dpappas/dpappas_data/models_out/ablations/test_ablation_1111111_batch1/v3 test_emit_bioasq.json"

python \
/home/dpappas/bioasq_all/eval/run_eval.py \
"/home/dpappas/test_bert_jpdrmm/v3 test_gold_bioasq.json" \
"/home/dpappas/test_bert_jpdrmm/elk_relevant_abs_posit_drmm_lists_test.json" \
 | grep map

MAP documents   : 0.08401785714285709
MAP snippets    : 0.05846759009205377
GMAP documents  : 0.003971499747357504
GMAP snippets   : 9.80472859647476E-4
F1 snippets     : 0.09388342171746662

trec map doc    : 0.4252

python3.6 tt.py -30. -30. False
grep -E '\"body\"|\"text\"' "test_bert_jpdrmm_high_batch3/v3 test_emit_bioasq.json"
cp "/home/dpappas/test_bert_jpdrmm_high_batch3/v3 test_emit_bioasq.json" "/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/bert_jpdrmm.json" 

CUDA_VISIBLE_DEVICES=1 python3.6 extract_bert_jpdrmm.py 1 &
CUDA_VISIBLE_DEVICES=0 python3.6 extract_bert_jpdrmm.py 2 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_bert_jpdrmm.py 3 &
CUDA_VISIBLE_DEVICES=0 python3.6 extract_bert_jpdrmm.py 4 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_bert_jpdrmm.py 5

java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' \
evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_5/BioASQ-task7bPhaseB-testset5" \
"./test_bert_jpdrmm_high_batch5/v3 test_emit_bioasq.json"

/media/dpappas/dpappas_data/models_out/bioasq7_bert_jpdrmm_2L_0p01_unfrozen_run_0/best_checkpoint.pth.tar

java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/test_bert_jpdrmm_frozen_high_batch1/v3 test_emit_bioasq.json"

'''



