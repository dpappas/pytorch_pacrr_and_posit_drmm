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
from    pytorch_transformers import BertModel, BertTokenizer

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
  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

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

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens          = []
        input_type_ids  = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        #
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)
        #
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #
        input_mask = [1] * len(input_ids)
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id       = example.guid,
                tokens          = tokens,
                input_ids       = input_ids,
                input_mask      = input_mask,
                input_type_ids  = input_type_ids
            )
        )
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

def get_embeds_use_unk(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if (tok in wv):
            ret1.append(tok)
            ret2.append(wv[tok])
        else:
            wv[tok] = np.random.randn(embedding_dim)
            ret1.append(tok)
            ret2.append(wv[tok])
    return ret1, np.array(ret2, 'float64')

def get_embeds_use_only_unk(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        wv[tok] = np.random.randn(embedding_dim)
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
            dtext = doc_text[doc_id]['title'] + ' <title> ' + doc_text[doc_id]['abstractText']
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
    norm_doc_scores     = get_norm_doc_scores(the_doc_scores)
    # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
    extracted_snippets  = [tt for tt in extracted_snippets if (tt[2] in norm_doc_scores)]
    sorted_snips        = sorted(extracted_snippets, key=lambda x: x[1] * norm_doc_scores[x[2]], reverse=True)
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

def get_pseudo_retrieved(dato, bioasq6_data):
    some_ids = [item['document'].split('/')[-1].strip() for item in bioasq6_data[dato['query_id']]['snippets']]
    pseudo_retrieved = [
        {
            'bm25_score': 7.76,
            'doc_id': id,
            'is_relevant': True,
            'norm_bm25_score': 3.85
        }
        for id in set(some_ids)
    ]
    return pseudo_retrieved

def get_snippets_loss(good_sent_tags, gs_emits_, bs_emits_):
    wright = torch.cat([gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 1)])
    wrong = [gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 0)]
    wrong = torch.cat(wrong + [bs_emits_.squeeze(-1)])
    losses = [model.my_hinge_loss(w.unsqueeze(0).expand_as(wrong), wrong) for w in wright]
    return sum(losses) / float(len(losses))

def get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_):
    bs_emits_       = bs_emits_.squeeze(-1)
    gs_emits_       = gs_emits_.squeeze(-1)
    good_sent_tags  = torch.FloatTensor(good_sent_tags).to(device)
    tags_2          = torch.zeros_like(bs_emits_).to(device)
    #
    sn_d1_l = F.binary_cross_entropy(gs_emits_, good_sent_tags, reduction='sum')
    sn_d2_l = F.binary_cross_entropy(bs_emits_, tags_2, reduction='sum')
    return sn_d1_l, sn_d2_l

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

def embed_the_sents(sents, questions):
    eval_examples       = []
    c = 0
    for sent, question in zip(sents, questions):
        eval_examples.append(InputExample(guid='example_dato_{}'.format(str(c)), text_a=sent, text_b=question, label=str(c)))
        c+=1
    eval_features       = convert_examples_to_features(eval_examples, 256, bert_tokenizer)
    input_ids           = torch.tensor([ef.input_ids for ef in eval_features], dtype=torch.long).to(device)
    attention_mask      = torch.tensor([ef.input_mask for ef in eval_features], dtype=torch.long).to(device)
    with torch.no_grad():
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
        head_mask               = [None] * bert_model.config.num_hidden_layers
        token_type_ids          = torch.zeros_like(input_ids).to(device)
        embedding_output        = bert_model.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids)
        sequence_output, rest   = bert_model.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
        if(adapt):
            first_token_tensors     = torch.stack([r[:, 0, :] for r in rest], dim=-1)
            weighted_vecs           = torch.matmul(first_token_tensors, layers_weights).squeeze(-1)
        else:
            weighted_vecs           = sequence_output[:, 0, :]
    return weighted_vecs

def get_map_res(fgold, femit, eval_path):
    print('Command: '+ ' '.join(['python', eval_path, fgold, femit]))
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
    emitss  = gs_emits_.squeeze().tolist()
    mmax    = max(emitss)
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
    good_sents_escores, held_out_sents, good_sent_tags = [], [], []
    for good_text in good_sents:
        sent_toks               = bioclean(good_text)
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
        good_sents_escores.append(good_escores + features)
        held_out_sents.append(good_text)
        good_sent_tags.append(snip_is_relevant(' '.join(bioclean(good_text)), good_snips))
    ####
    sents_embeds = embed_the_sents(held_out_sents, [quest] * len(held_out_sents))
    ####
    return {
        'sents_embeds'  : sents_embeds,
        'sents_escores' : good_sents_escores,
        'doc_af'        : good_doc_af,
        'sent_tags'     : good_sent_tags,
        'held_out_sents': held_out_sents
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
    quest_tokens        = quest_text.split()
    ####
    gold_snips          = []
    #
    doc_res, extracted_snippets         = {}, []
    extracted_snippets_known_rel_num    = []
    for retr in retr_docs:
        datum                   = prep_data(quest_text, docs[retr['doc_id']], retr['norm_bm25_score'], gold_snips, quest_tokens)
        doc_emit_, gs_emits_    = model.emit_one(
            doc1_sents_embeds   = datum['sents_embeds'],
            doc1_saf            = datum['sents_escores'],
            doc1_daf            = datum['doc_af']
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
    snips_res_known_rel_num_v3 = prep_extracted_snippets(extracted_snippets_known_rel_num_v3, docs, dato['query_id'], doc_res[:10], dato['query_text'])
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
    ###########################################################
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
    ###########################################################

def train_data_step1(train_data):
    ret = []
    for dato in tqdm(train_data['queries'], ascii=True):
        quest = dato['query_text']
        quest_id = dato['query_id']
        bm25s = {t['doc_id']: t['norm_bm25_score'] for t in dato[u'retrieved_documents']}
        ret_pmids = [t[u'doc_id'] for t in dato[u'retrieved_documents']]
        good_pmids = [t for t in ret_pmids if t in dato[u'relevant_documents']]
        bad_pmids = [t for t in ret_pmids if t not in dato[u'relevant_documents']]
        if (len(bad_pmids) > 0):
            for gid in good_pmids:
                bid = random.choice(bad_pmids)
                ret.append((quest, quest_id, gid, bid, bm25s[gid], bm25s[bid]))
    print('')
    return ret

def train_data_step2(instances, docs, bioasq6_data, use_sent_tokenizer):
    for quest_text, quest_id, gid, bid, bm25s_gid, bm25s_bid in instances:
        ####
        good_snips          = get_snips(quest_id, gid, bioasq6_data)
        good_snips          = [' '.join(bioclean(sn)) for sn in good_snips]
        quest_text          = ' '.join(bioclean(quest_text.replace('\ufeff', ' ')))
        quest_tokens        = quest_text.split()
        ####
        datum               = prep_data(quest_text, docs[gid], bm25s_gid, good_snips, quest_tokens)
        good_sents_embeds   = datum['sents_embeds']
        good_sents_escores  = datum['sents_escores']
        good_doc_af         = datum['doc_af']
        good_sent_tags      = datum['sent_tags']
        good_held_out_sents = datum['held_out_sents']
        #
        datum               = prep_data(quest_text, docs[bid], bm25s_bid, [], quest_tokens)
        bad_sents_embeds    = datum['sents_embeds']
        bad_sents_escores   = datum['sents_escores']
        bad_doc_af          = datum['doc_af']
        bad_sent_tags       = [0] * len(datum['sent_tags'])
        bad_held_out_sents  = datum['held_out_sents']
        #
        yield {
                'good_sents_embeds'     : good_sents_embeds,
                'good_sents_escores'    : good_sents_escores,
                'good_doc_af'           : good_doc_af,
                'good_sent_tags'        : good_sent_tags,
                'good_held_out_sents'   : good_held_out_sents,
                #
                'bad_sents_embeds'      : bad_sents_embeds,
                'bad_sents_escores'     : bad_sents_escores,
                'bad_doc_af'            : bad_doc_af,
                'bad_sent_tags'         : bad_sent_tags,
                'bad_held_out_sents'    : bad_held_out_sents
                #
            }

def get_one_map(prefix, data, docs, use_sent_tokenizer):
    model.eval()
    bert_model.eval()
    #
    ret_data = {'questions': []}
    all_bioasq_subm_data_v3 = {"questions": []}
    all_bioasq_subm_data_known_v3 = {"questions": []}
    all_bioasq_gold_data = {'questions': []}
    data_for_revision = {}
    #
    for dato in tqdm(data['queries'], ascii=True):
        all_bioasq_gold_data['questions'].append(bioasq7_data[dato['query_id']])
        data_for_revision, ret_data, snips_res, snips_res_known = do_for_some_retrieved(
            docs, dato, dato['retrieved_documents'], data_for_revision, ret_data, use_sent_tokenizer)
        all_bioasq_subm_data_v3['questions'].append(snips_res['v3'])
        all_bioasq_subm_data_known_v3['questions'].append(snips_res_known['v3'])
    #
    print_the_results('v3 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v3, all_bioasq_subm_data_known_v3, data_for_revision)
    #
    if (prefix == 'dev'):
        with open(os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(
            os.path.join(odir, 'v3 dev_gold_bioasq.json'),
            os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json'),
            eval_path
        )
    else:
        with open(os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_test.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(
            os.path.join(odir, 'v3 test_emit_bioasq.json'),
            os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_test.json'),
            eval_path
        )
    return res_map

class JBERT(nn.Module):
    def __init__(self, embedding_dim=768, k_for_maxpool=5, k_sent_maxpool=1):
        super(JBERT, self).__init__()
        self.k                      = k_for_maxpool
        self.k_sent_maxpool         = k_sent_maxpool
        self.doc_add_feats          = 11
        self.sent_add_feats         = 10
        self.embedding_dim          = embedding_dim
        ##########################
        self.sentence_scorer_0      = nn.Linear(self.embedding_dim, 8)
        self.sentence_scorer_1      = nn.Linear(8, 1)
        self.sentence_scorer_2      = nn.Linear(1+self.sent_add_feats, 1)
        ##########################
        self.doc_scorer_0           = nn.Linear(1+self.doc_add_feats, 8)
        self.doc_scorer_1           = nn.Linear(8, 1)
        ##########################
    #
    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos
    #
    def emit_one(self, doc1_sents_embeds, doc1_saf, doc1_daf):
        doc1_saf = autograd.Variable(torch.FloatTensor(doc1_saf), requires_grad=False).to(device)
        doc1_daf = autograd.Variable(torch.FloatTensor(doc1_daf), requires_grad=False).to(device)
        ################################################################
        doc1_sent_scores        = torch.tanh(self.sentence_scorer_0(doc1_sents_embeds))
        doc1_sent_scores        = torch.sigmoid(self.sentence_scorer_1(doc1_sent_scores))
        doc1_sent_scores        = torch.cat((doc1_sent_scores, doc1_saf), dim=-1)
        doc1_sent_scores        = torch.sigmoid(self.sentence_scorer_2(doc1_sent_scores))
        ###############################################################
        doc1_sent_max_score     = doc1_sent_scores.max()
        doc1_doc_score          = torch.cat((doc1_sent_max_score.unsqueeze(0), doc1_daf))
        doc1_doc_score          = F.leaky_relu(self.doc_scorer_0(doc1_doc_score))
        doc1_doc_score          = self.doc_scorer_1(doc1_doc_score)
        return doc1_doc_score, doc1_sent_scores
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, doc1_saf, doc2_saf, doc1_daf, doc2_daf):
        doc1_saf = autograd.Variable(torch.FloatTensor(doc1_saf), requires_grad=False).to(device)
        doc2_saf = autograd.Variable(torch.FloatTensor(doc2_saf), requires_grad=False).to(device)
        doc1_daf = autograd.Variable(torch.FloatTensor(doc1_daf), requires_grad=False).to(device)
        doc2_daf = autograd.Variable(torch.FloatTensor(doc2_daf), requires_grad=False).to(device)
        ################################################################
        doc1_sent_scores        = torch.tanh(self.sentence_scorer_0(doc1_sents_embeds))
        doc1_sent_scores        = torch.sigmoid(self.sentence_scorer_1(doc1_sent_scores))
        doc1_sent_scores        = torch.cat((doc1_sent_scores, doc1_saf), dim=-1)
        doc1_sent_scores        = torch.sigmoid(self.sentence_scorer_2(doc1_sent_scores))
        ###############################################################
        doc1_sent_max_score     = doc1_sent_scores.max()
        doc1_doc_score          = torch.cat((doc1_sent_max_score.unsqueeze(0), doc1_daf))
        doc1_doc_score          = F.leaky_relu(self.doc_scorer_0(doc1_doc_score))
        doc1_doc_score          = self.doc_scorer_1(doc1_doc_score)
        ###############################################################
        doc2_sent_scores        = torch.tanh(self.sentence_scorer_0(doc2_sents_embeds))
        doc2_sent_scores        = torch.sigmoid(self.sentence_scorer_1(doc2_sent_scores))
        doc2_sent_scores        = torch.cat((doc2_sent_scores, doc2_saf), dim=-1)
        doc2_sent_scores        = torch.sigmoid(self.sentence_scorer_2(doc2_sent_scores))
        ###############################################################
        doc2_sent_max_score     = doc2_sent_scores.max()
        doc2_doc_score          = torch.cat((doc2_sent_max_score.unsqueeze(0), doc2_daf))
        doc2_doc_score          = F.leaky_relu(self.doc_scorer_0(doc2_doc_score))
        doc2_doc_score          = self.doc_scorer_1(doc2_doc_score)
        ###############################################################
        loss1                   = self.my_hinge_loss(doc1_doc_score, doc2_doc_score)
        return loss1, doc1_doc_score, doc2_doc_score, doc1_sent_scores, doc2_sent_scores

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

###########################################################
use_cuda            = torch.cuda.is_available()
###########################################################
batch_no            = sys.argv[1]
f_in1               = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseA-testset{}'.format(batch_no, batch_no)
f_in2               = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'.format(batch_no)
f_in3               = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl'.format(batch_no)
###########################################################
resume_from         = '/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_run_frozen/'
odir                = os.path.join(resume_from, 'batch_{}'.format(batch_no))
adapt               = '_adapt_' in resume_from
if(adapt):
    layers_weights  = None
else:
    layers_weights  = None
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
embedding_dim       = 768 #200
###########################################################
my_seed     = 1
random.seed(my_seed)
torch.manual_seed(my_seed)
###########################################################
use_cuda            = True
device              = torch.device("cuda") if(use_cuda) else torch.device("cpu")
###########################################################
model               = JBERT(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool).to(device)
#####################
cache_dir           = 'bert-base-uncased' # '/home/dpappas/bert_cache/'
bert_tokenizer      = BertTokenizer.from_pretrained(cache_dir)
bert_model          = BertModel.from_pretrained(cache_dir,  output_hidden_states=True, output_attentions=False).to(device)
load_model_from_checkpoint(resume_from)
for param in model.parameters():
    param.requires_grad = False
for param in bert_model.parameters():
    param.requires_grad = False
###########################################################
# print_params(model)
# print_params(bert_model)
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
###########################################################

# CUDA_VISIBLE_DEVICES=1 python3.6 extract_jbert_adapt_nf.py 1

'''
python "/home/dpappas/bioasq_all/eval/run_eval.py" \
"/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_run_frozen/batch_1/v3 test_emit_bioasq.json" \
"/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_run_frozen/batch_1/elk_relevant_abs_posit_drmm_lists_test.json"
'''

'''
java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_run_frozen/batch_1/v3 test_emit_bioasq.json" \
| grep "^MAP documents:\|^MAP snippets:"


java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_run_frozen/batch_2/v3 test_emit_bioasq.json" \
| grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_run_frozen/batch_3/v3 test_emit_bioasq.json" \
| grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_4/BioASQ-task7bPhaseB-testset4" \
"/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_run_frozen/batch_4/v3 test_emit_bioasq.json" \
| grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_5/BioASQ-task7bPhaseB-testset5" \
"/media/dpappas/dpappas_data/models_out/bioasq7_jbertadaptnf_toponly_run_frozen/batch_5/v3 test_emit_bioasq.json" \
| grep "^MAP documents:\|^MAP snippets:"

'''
