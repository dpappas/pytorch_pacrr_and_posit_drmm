
import  torch, pickle, os, re, nltk, logging
import  torch.nn.functional     as F
import  numpy                   as np
from    nltk.tokenize           import sent_tokenize
from    tqdm                    import tqdm
from    sklearn.preprocessing   import LabelEncoder
from    sklearn.preprocessing   import OneHotEncoder
from    difflib                 import SequenceMatcher

softmax     = lambda z: np.exp(z) / np.sum(np.exp(z))
stopwords   = nltk.corpus.stopwords.words("english")
bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

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

def init_the_logger(hdlr, odir):
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

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
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
        input_ids       = tokenizer.convert_tokens_to_ids(tokens)
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

def get_words(s, idf, max_idf):
    sl = tokenize(s)
    sl = [s for s in sl]
    sl2 = [s for s in sl if idf_val(s, idf, max_idf) >= 2.0]
    return sl, sl2

def fix_bert_tokens(tokens):
    ret = []
    for t in tokens:
        if (t.startswith('##')):
            ret[-1] = ret[-1] + t[2:]
        else:
            ret.append(t)
    return ret

def tokenize(x, bert_tokenizer):
    x_tokens = bert_tokenizer.tokenize(x)
    x_tokens = fix_bert_tokens(x_tokens)
    return x_tokens

def idf_val(w, idf, max_idf):
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

def get_embeds_use_unk(tokens, wv, embedding_dim):
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

def get_embeds_use_only_unk(tokens, wv, embedding_dim):
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

def query_doc_overlap(qwords, dwords, idf, max_idf):
    # % Query words in doc.
    qwords_in_doc = 0
    idf_qwords_in_doc = 0.0
    idf_qwords = 0.0
    for qword in uwords(qwords):
        idf_qwords += idf_val(qword, idf, max_idf)
        for dword in uwords(dwords):
            if qword == dword:
                idf_qwords_in_doc += idf_val(qword, idf, max_idf)
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
        idf_bigrams += idf_val(wrds[0], idf, max_idf) * idf_val(wrds[1], idf, max_idf)
        for dword in ubigrams(dwords):
            if qword == dword:
                qwords_bigrams_in_doc += 1
                idf_qwords_bigrams_in_doc += (idf_val(wrds[0], idf, max_idf) * idf_val(wrds[1], idf, max_idf))
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

def GetScores(qtext, dtext, bm25, idf, max_idf):
    qwords, qw2 = get_words(qtext, idf, max_idf)
    dwords, dw2 = get_words(dtext, idf, max_idf)
    qd1 = query_doc_overlap(qwords, dwords, idf, max_idf)
    bm25 = [bm25]
    return qd1[0:3] + bm25

def GetWords(data, doc_text, words):
    for i in tqdm(range(len(data['queries'])), ascii=True):
        qwds = tokenize(data['queries'][i]['query_text'])
        for w in qwds:
            words[w] = 1
        for j in range(len(data['queries'][i]['retrieved_documents'])):
            doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
            dtext = (
                    doc_text[doc_id]['title'] + ' <title> ' + doc_text[doc_id]['abstractText'] +
                    ' '.join(
                        [
                            ' '.join(mm) for mm in
                            get_the_mesh(doc_text[doc_id])
                        ]
                    )
            )
            dwds = tokenize(dtext)
            for w in dwds:
                words[w] = 1

def get_gold_snips(quest_id, bioasq6_data):
    gold_snips = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            gold_snips.extend(sent_tokenize(sn['text']))
    return list(set(gold_snips))

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

def get_the_mesh(the_doc):
    good_meshes = []
    if ('meshHeadingsList' in the_doc):
        for t in the_doc['meshHeadingsList']:
            t = t.split(':', 1)
            t = t[1].strip()
            t = t.lower()
            good_meshes.append(t)
    elif ('MeshHeadings' in the_doc):
        for mesh_head_set in the_doc['MeshHeadings']:
            for item in mesh_head_set:
                good_meshes.append(item['text'].strip().lower())
    if ('Chemicals' in the_doc):
        for t in the_doc['Chemicals']:
            t = t['NameOfSubstance'].strip().lower()
            good_meshes.append(t)
    good_mesh = sorted(good_meshes)
    good_mesh = ['mesh'] + good_mesh
    # good_mesh = ' # '.join(good_mesh)
    # good_mesh = good_mesh.split()
    # good_mesh = [gm.split() for gm in good_mesh]
    good_mesh = [gm for gm in good_mesh]
    return good_mesh

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

def select_snippets_v1(extracted_snippets):
    '''
    :param extracted_snippets:
    :param doc_res:
    :return: returns the best 10 snippets of all docs (0..n from each doc)
    '''
    sorted_snips = sorted(extracted_snippets, key=lambda x: x[1], reverse=True)
    return sorted_snips[:10]

def select_snippets_v2(extracted_snippets):
    '''
    :param extracted_snippets:
    :param doc_res:
    :return: returns the best snippet of each doc  (1 from each doc)
    '''
    # is_relevant, the_sent_score, ncbi_pmid_link, the_actual_sent_text
    ret = {}
    for es in extracted_snippets:
        if (es[2] in ret):
            if (es[1] > ret[es[2]][1]):
                ret[es[2]] = es
        else:
            ret[es[2]] = es
    sorted_snips = sorted(ret.values(), key=lambda x: x[1], reverse=True)
    return sorted_snips[:10]

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

def get_snippets_loss(good_sent_tags, gs_emits_, bs_emits_, model):
    wright = torch.cat([gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 1)])
    wrong = [gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 0)]
    wrong = torch.cat(wrong + [bs_emits_.squeeze(-1)])
    losses = [model.my_hinge_loss(w.unsqueeze(0).expand_as(wrong), wrong) for w in wright]
    return sum(losses) / float(len(losses))

def get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_, device):
    bs_emits_ = bs_emits_.squeeze(-1)
    gs_emits_ = gs_emits_.squeeze(-1)
    good_sent_tags = torch.FloatTensor(good_sent_tags).to(device)
    tags_2 = torch.zeros_like(bs_emits_).to(device)
    #
    sn_d1_l = F.binary_cross_entropy(gs_emits_, good_sent_tags, reduction='sum')
    sn_d2_l = F.binary_cross_entropy(bs_emits_, tags_2, reduction='sum')
    return sn_d1_l, sn_d2_l



