
import  random, os, pickle, nltk, re, json, logging, math, time, subprocess
import  torch
from    pytorch_transformers    import *
from    torch                   import nn
from    torch.nn                import CrossEntropyLoss, MSELoss
import  torch.nn.functional     as F
import  torch.optim             as optim
from    tqdm                    import tqdm
from    pprint                  import pprint
import  numpy                   as np
from    nltk.tokenize           import sent_tokenize
import  torch.autograd              as autograd

softmax     = lambda z: np.exp(z) / np.sum(np.exp(z))
stopwords   = nltk.corpus.stopwords.words("english")
bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

my_seed = 1
random.seed(my_seed)
torch.manual_seed(my_seed)

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

def get_bm25_metrics(avgdl=0., mean=0., deviation=0.):
    if (avgdl == 0):
        total_words = 0
        total_docs = 0
        for dic in tqdm(train_docs, ascii=True):
            sents = sent_tokenize(train_docs[dic]['title']) + sent_tokenize(train_docs[dic]['abstractText'])
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
            qtext = bioasq6_data[qid]['body']
            all_retr_ids = [link.split('/')[-1] for link in bioasq6_data[qid]['documents']]
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

def tokenize(x):
    x_tokens = bert_tokenizer.tokenize(x)
    x_tokens = fix_bert_tokens(x_tokens)
    return x_tokens

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

def save_checkpoint(epoch, model, max_dev_map, optimizer, scheduler, filename='checkpoint.pth.tar'):
    '''
    :param state:       the stete of the pytorch mode
    :param filename:    the name of the file in which we will store the model.
    :return:            Nothing. It just saves the model.
    '''
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_valid_score': max_dev_map,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, filename)

def encode_sents_faster_but_padding(sents):
    ###############################################################
    tokenized_sents = [bert_tokenizer.encode(sent) for sent in sents]
    max_len         = max(len(sent) for sent in tokenized_sents)
    pad_id          = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.pad_token)
    input_ids       = torch.tensor([sent_ids + ([pad_id] * (max_len - len(sent_ids))) for sent_ids in tokenized_sents])
    ###############################################################
    with torch.no_grad():
        last_hidden_state, pooler_output, hidden_states, attentions = bert_model(input_ids)
    ###############################################################
    # print(last_hidden_state.size())
    # print(pooler_output.size())
    # print(len(hidden_states))
    # print([t.size() for t in hidden_states])
    # print(len(attentions))
    # print([t.size() for t in attentions])
    ###############################################################
    return last_hidden_state, pooler_output, hidden_states, attentions

def my_hinge_loss(positives, negatives, margin=1.0):
    delta = negatives - positives
    loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
    return loss_q_pos

def encode_sents(sents):
    try:
        last_hidden_state, pooler_output = [], []
        for sent in sents:
            tokenized_sent = bert_tokenizer.encode(sent, add_special_tokens=True)
            input_ids       = torch.tensor([tokenized_sent])
            with torch.no_grad():
                _last_hidden_state, _pooler_output = bert_model(input_ids)
                last_hidden_state.append(_last_hidden_state)
                pooler_output.append(_pooler_output)
        return last_hidden_state, torch.cat(pooler_output, dim=0)
    except:
        print(20 * '-')
        pprint(sents)
        return None, None

def print_params(model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(model)
    print(40 * '=')
    trainable = 0
    untrainable = 0
    # for parameter in list(model.parameters()) + list(bert_model.parameters()):
    for parameter in list(model.parameters()):
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        if (parameter.requires_grad):
            trainable += v
        else:
            untrainable += v
    total_params = trainable + untrainable
    print(40 * '=')
    print('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    print(40 * '=')

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

def load_all_data(dataloc, idf_pickle_path):
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

def prep_data(quest, the_doc, the_bm25, good_snips, idf, max_idf):
    good_sents          = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
    quest_toks          = bioclean(quest)
    ####
    good_doc_af         = GetScores(quest, the_doc['title'] + the_doc['abstractText'], the_bm25, idf, max_idf)
    good_doc_af.append(len(good_sents) / 60.)
    #
    all_doc_text        = the_doc['title'] + ' ' + the_doc['abstractText']
    doc_toks            = tokenize(all_doc_text)
    tomi                = (set(doc_toks) & set(quest_toks))
    tomi_no_stop        = tomi - set(stopwords)
    BM25score           = similarity_score(quest_toks, doc_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
    tomi_no_stop_idfs   = [idf_val(w, idf, max_idf) for w in tomi_no_stop]
    tomi_idfs           = [idf_val(w, idf, max_idf) for w in tomi]
    quest_idfs          = [idf_val(w, idf, max_idf) for w in quest_toks]
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
    sents_to_embed = []
    for good_text in good_sents:
        sent_toks               = bioclean(good_text)
        sents_to_embed.append(' '.join(['PAD']+sent_toks))
        # sent_embeds             = torch.cat([sent_embeds_1, sent_embeds_2])
        good_escores            = GetScores(quest, good_text, the_bm25, idf, max_idf)[:-1]
        good_escores.append(len(sent_toks) / 342.)
        tomi                    = (set(sent_toks) & set(quest_toks))
        tomi_no_stop            = tomi - set(stopwords)
        BM25score               = similarity_score(quest_toks, sent_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
        tomi_no_stop_idfs       = [idf_val(w, idf, max_idf) for w in tomi_no_stop]
        tomi_idfs               = [idf_val(w, idf, max_idf) for w in tomi]
        quest_idfs              = [idf_val(w, idf, max_idf) for w in quest_toks]
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
    _, good_sents_embeds = encode_sents(sents_to_embed)
    ####
    return {
        'sents_embeds'  : good_sents_embeds,
        'sents_escores' : good_sents_escores,
        'doc_af'        : good_doc_af,
        'sent_tags'     : good_sent_tags,
        'held_out_sents': held_out_sents
    }

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

def train_data_step2(instances, docs, bioasq6_data, idf, max_idf, use_sent_tokenizer):
    for quest_text, quest_id, gid, bid, bm25s_gid, bm25s_bid in instances:
        ####
        good_snips          = get_snips(quest_id, gid, bioasq6_data)
        good_snips          = [' '.join(bioclean(sn)) for sn in good_snips]
        quest_text          = ' '.join(bioclean(quest_text.replace('\ufeff', ' ')))
        quest_tokens        = bioclean(quest_text)
        ####
        datum               = prep_data(quest_text, docs[gid], bm25s_gid, good_snips, idf, max_idf)
        good_sents_embeds   = datum['sents_embeds']
        good_sents_escores  = datum['sents_escores']
        good_doc_af         = datum['doc_af']
        good_sent_tags      = datum['sent_tags']
        good_held_out_sents = datum['held_out_sents']
        #
        datum               = prep_data(quest_text, docs[bid], bm25s_bid, [], idf, max_idf)
        bad_sents_embeds    = datum['sents_embeds']
        bad_sents_escores   = datum['sents_escores']
        bad_doc_af          = datum['doc_af']
        bad_sent_tags       = [0] * len(datum['sent_tags'])
        bad_held_out_sents  = datum['held_out_sents']
        #
        if (use_sent_tokenizer == False or sum(good_sent_tags) > 0):
            yield {
                'good_sents_embeds': good_sents_embeds,
                'good_sents_escores': good_sents_escores,
                'good_doc_af': good_doc_af,
                'good_sent_tags': good_sent_tags,
                'good_held_out_sents': good_held_out_sents,
                #
                'bad_sents_embeds': bad_sents_embeds,
                'bad_sents_escores': bad_sents_escores,
                'bad_doc_af': bad_doc_af,
                'bad_sent_tags': bad_sent_tags,
                'bad_held_out_sents': bad_held_out_sents,
            }

def get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_):
    bs_emits_       = bs_emits_.squeeze(-1)
    gs_emits_       = gs_emits_.squeeze(-1)
    good_sent_tags  = torch.FloatTensor(good_sent_tags).to(device)
    tags_2          = torch.zeros_like(bs_emits_).to(device)
    # print(gs_emits_)
    # print(good_sent_tags)
    #
    # sn_d1_l = F.binary_cross_entropy(gs_emits_, good_sent_tags, size_average=False, reduce=True)
    # sn_d2_l = F.binary_cross_entropy(bs_emits_, tags_2, size_average=False, reduce=True)
    # print(gs_emits_)
    # print(good_sent_tags)
    # print(bs_emits_)
    # print(tags_2)
    sn_d1_l = F.binary_cross_entropy(gs_emits_, good_sent_tags, reduction='sum')
    sn_d2_l = F.binary_cross_entropy(bs_emits_, tags_2, reduction='sum')
    return sn_d1_l, sn_d2_l

def back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc):
    batch_cost = sum(batch_costs) / float(len(batch_costs))
    batch_cost.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    model.zero_grad()
    batch_aver_cost = batch_cost.cpu().item()
    epoch_aver_cost = sum(epoch_costs) / float(len(epoch_costs))
    batch_aver_acc = sum(batch_acc) / float(len(batch_acc))
    epoch_aver_acc = sum(epoch_acc) / float(len(epoch_acc))
    return batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc

def train_one(epoch, bioasq6_data, two_losses, use_sent_tokenizer):
    model.train()
    bert_model.train()
    batch_costs, batch_acc, epoch_costs, epoch_acc = [], [], [], []
    batch_counter, epoch_aver_cost, epoch_aver_acc = 0, 0., 0.
    #
    train_instances = train_data_step1(train_data)
    random.shuffle(train_instances)
    #
    start_time = time.time()
    pbar = tqdm(
        iterable= train_data_step2(train_instances, train_docs, bioasq6_data, idf, max_idf, use_sent_tokenizer),
        total   = 14288, #9684, # 378,
        ascii   = True
    )
    for datum in pbar:
        doc1_emit_, gs_emits_, doc2_emit_, bs_emits_ = model(
            datum['good_sents_embeds'].to(device),
            datum['bad_sents_embeds'].to(device),
            datum['good_sents_escores'],
            datum['bad_sents_escores'],
            datum['good_doc_af'],
            datum['bad_doc_af']
        )
        #
        cost_ = my_hinge_loss(doc1_emit_, doc2_emit_)
        #
        good_sent_tags, bad_sent_tags = datum['good_sent_tags'], datum['bad_sent_tags']
        if (two_losses):
            sn_d1_l, sn_d2_l = get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_)
            snip_loss = sn_d1_l + sn_d2_l
            l = 0.5
            cost_ = ((1 - l) * snip_loss) + (l * cost_)
        #
        batch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_costs.append(cost_.cpu().item())
        batch_costs.append(cost_)
        if (len(batch_costs) == b_size):
            batch_counter += 1
            batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(
                batch_costs, epoch_costs, batch_acc, epoch_acc)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                batch_counter, batch_aver_cost, epoch_aver_cost,batch_aver_acc, epoch_aver_acc, elapsed_time)
            )
            logger.info(
                '{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                    batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc, elapsed_time)
            )
            batch_costs, batch_acc = [], []
    if (len(batch_costs) > 0):
        batch_counter += 1
        batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs,
                                                                                     batch_acc, epoch_acc)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                 batch_aver_acc, epoch_aver_acc, elapsed_time))
        logger.info('{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                       batch_aver_acc, epoch_aver_acc, elapsed_time))
    print('Epoch:{:02d} aver_epoch_cost: {:.4f} aver_epoch_acc: {:.4f}'.format(epoch, epoch_aver_cost, epoch_aver_acc))
    logger.info(
        'Epoch:{:02d} aver_epoch_cost: {:.4f} aver_epoch_acc: {:.4f}'.format(epoch, epoch_aver_cost, epoch_aver_acc))

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

def print_the_results(prefix, all_bioasq_gold_data, all_bioasq_subm_data, all_bioasq_subm_data_known, data_for_revision):
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data_known, data_for_revision)
    pprint(bioasq_snip_res)
    print('{} known MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    print('{} known F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    print('{} known MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    print('{} known GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    logger.info('{} known MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    logger.info('{} known F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    logger.info('{} known MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    logger.info('{} known GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data, data_for_revision)
    pprint(bioasq_snip_res)
    print('{} MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    print('{} F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    print('{} MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    print('{} GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    logger.info('{} MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    logger.info('{} F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    logger.info('{} MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    logger.info('{} GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #

def get_map_res(fgold, femit):
    trec_eval_res = subprocess.Popen(['python', eval_path, fgold, femit], stdout=subprocess.PIPE, shell=False)
    (out, err) = trec_eval_res.communicate()
    lines = out.decode("utf-8").split('\n')
    map_res = [l for l in lines if (l.startswith('map '))][0].split('\t')
    map_res = float(map_res[-1])
    return map_res

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

def do_for_one_retrieved(doc_emit_, gs_emits_, held_out_sents, retr, doc_res, gold_snips):
    emition = doc_emit_.cpu().item()
    emitss  = gs_emits_.tolist()
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

def do_for_some_retrieved(docs, dato, retr_docs, data_for_revision, ret_data, use_sent_tokenizer):
    emitions = {
        'body': dato['query_text'],
        'id': dato['query_id'],
        'documents': []
    }
    ####
    quest_text          = dato['query_text']
    quest_text          = ' '.join(bioclean(quest_text.replace('\ufeff', ' ')))
    quest_tokens        = bioclean(quest_text)
    ####
    gold_snips          = get_gold_snips(dato['query_id'], bioasq6_data)
    #
    doc_res, extracted_snippets         = {}, []
    extracted_snippets_known_rel_num    = []
    for retr in retr_docs:
        datum                   = prep_data(quest_text, docs[retr['doc_id']], retr['norm_bm25_score'], gold_snips, idf, max_idf)
        doc_emit_, gs_emits_ = model.emit_one(
            doc1_sents_embeds   = datum['sents_embeds'].to(device),
            doc1_sents_af       = datum['sents_escores'],
            doc1_af             = datum['doc_af']
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
        extracted_snippets_v1 = select_snippets_v1(extracted_snippets)
        extracted_snippets_v2 = select_snippets_v2(extracted_snippets)
        # pprint(extracted_snippets[:20])
        extracted_snippets_v3 = select_snippets_v3(extracted_snippets, the_doc_scores)
        extracted_snippets_known_rel_num_v1 = select_snippets_v1(extracted_snippets_known_rel_num)
        extracted_snippets_known_rel_num_v2 = select_snippets_v2(extracted_snippets_known_rel_num)
        extracted_snippets_known_rel_num_v3 = select_snippets_v3(extracted_snippets_known_rel_num, the_doc_scores)
    else:
        extracted_snippets_v1, extracted_snippets_v2, extracted_snippets_v3 = [], [], []
        extracted_snippets_known_rel_num_v1, extracted_snippets_known_rel_num_v2, extracted_snippets_known_rel_num_v3 = [], [], []
    #
    # pprint(extracted_snippets_v1)
    # pprint(extracted_snippets_v2)
    # pprint(extracted_snippets_v3)
    # exit()
    snips_res_v1 = prep_extracted_snippets(extracted_snippets_v1, docs, dato['query_id'], doc_res[:10],
                                           dato['query_text'])
    snips_res_v2 = prep_extracted_snippets(extracted_snippets_v2, docs, dato['query_id'], doc_res[:10],
                                           dato['query_text'])
    snips_res_v3 = prep_extracted_snippets(extracted_snippets_v3, docs, dato['query_id'], doc_res[:10],
                                           dato['query_text'])
    # pprint(snips_res_v1)
    # pprint(snips_res_v2)
    # pprint(snips_res_v3)
    # exit()
    #
    snips_res_known_rel_num_v1 = prep_extracted_snippets(extracted_snippets_known_rel_num_v1, docs, dato['query_id'],
                                                         doc_res[:10], dato['query_text'])
    snips_res_known_rel_num_v2 = prep_extracted_snippets(extracted_snippets_known_rel_num_v2, docs, dato['query_id'],
                                                         doc_res[:10], dato['query_text'])
    snips_res_known_rel_num_v3 = prep_extracted_snippets(extracted_snippets_known_rel_num_v3, docs, dato['query_id'],
                                                         doc_res[:10], dato['query_text'])
    #
    snips_res = {
        'v1': snips_res_v1,
        'v2': snips_res_v2,
        'v3': snips_res_v3,
    }
    snips_res_known = {
        'v1': snips_res_known_rel_num_v1,
        'v2': snips_res_known_rel_num_v2,
        'v3': snips_res_known_rel_num_v3,
    }
    return data_for_revision, ret_data, snips_res, snips_res_known

def get_one_map(prefix, data, docs, use_sent_tokenizer):
    model.eval()
    bert_model.eval()
    #
    ret_data = {'questions': []}
    all_bioasq_subm_data_v1 = {"questions": []}
    all_bioasq_subm_data_known_v1 = {"questions": []}
    all_bioasq_subm_data_v2 = {"questions": []}
    all_bioasq_subm_data_known_v2 = {"questions": []}
    all_bioasq_subm_data_v3 = {"questions": []}
    all_bioasq_subm_data_known_v3 = {"questions": []}
    all_bioasq_gold_data = {'questions': []}
    data_for_revision = {}
    #
    for dato in tqdm(data['queries'], ascii=True):
        all_bioasq_gold_data['questions'].append(bioasq6_data[dato['query_id']])
        data_for_revision, ret_data, snips_res, snips_res_known = do_for_some_retrieved(docs, dato, dato['retrieved_documents'], data_for_revision, ret_data, use_sent_tokenizer)
        all_bioasq_subm_data_v1['questions'].append(snips_res['v1'])
        all_bioasq_subm_data_v2['questions'].append(snips_res['v2'])
        all_bioasq_subm_data_v3['questions'].append(snips_res['v3'])
        all_bioasq_subm_data_known_v1['questions'].append(snips_res_known['v1'])
        all_bioasq_subm_data_known_v2['questions'].append(snips_res_known['v3'])
        all_bioasq_subm_data_known_v3['questions'].append(snips_res_known['v3'])
    #
    print_the_results('v1 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v1, all_bioasq_subm_data_known_v1, data_for_revision)
    print_the_results('v2 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v2, all_bioasq_subm_data_known_v2, data_for_revision)
    print_the_results('v3 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v3, all_bioasq_subm_data_known_v3, data_for_revision)
    #
    if (prefix == 'dev'):
        with open(os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(
            os.path.join(odir, 'v3 dev_gold_bioasq.json'),
            os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json')
        )
    else:
        with open(os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_test.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(
            os.path.join(odir, 'v3 test_gold_bioasq.json'),
            os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_test.json')
        )
    return res_map

class MLP(nn.Module):
    def __init__(self, input_dim=None, sizes=None, activation_functions=None, initializer_range=0.02):
        super(MLP, self).__init__()
        ################################
        sizes                   = [input_dim]+ sizes
        self.activations        = activation_functions
        self.linears            = nn.ModuleList()
        self.initializer_range  = initializer_range
        for i in range(len(sizes)-1):
            one_linear = nn.Linear(sizes[i], sizes[i+1])
            self.linears.append(one_linear)
        ################################
        self.apply(self.init_weights)
        for lin in self.linears:
            lin.apply(self.init_weights)
        ################################
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def forward(self, features):
        ret = features
        for layer, activation in zip(self.linears, self.activations):
            ret = layer(ret)
            if(activation is not None):
                ret = activation(ret)
        return ret

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias   = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class JBert(nn.Module):
    def __init__(self, embedding_dim=768, initializer_range=0.02):
        super(JBert, self).__init__()
        self.sent_add_feats     = 10
        self.doc_add_feats      = 11
        self.initializer_range  = initializer_range
        leaky_relu_lambda       = lambda t: F.leaky_relu(t, negative_slope=0.1)
        self.snip_MLP_1         = MLP(input_dim=embedding_dim,          sizes=[768, 1], activation_functions=[leaky_relu_lambda, torch.sigmoid]).to(device)
        self.snip_MLP_2         = MLP(input_dim=self.sent_add_feats+1,  sizes=[8, 1],   activation_functions=[leaky_relu_lambda, torch.sigmoid]).to(device)
        self.doc_MLP            = MLP(input_dim=self.doc_add_feats+1,   sizes=[8, 1],   activation_functions=[leaky_relu_lambda, leaky_relu_lambda]).to(device)
        self.apply(self.init_weights)
        #
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def emit_one(self, doc1_sents_embeds, doc1_sents_af, doc1_af):
        doc1_sents_af   = autograd.Variable(torch.FloatTensor(doc1_sents_af),   requires_grad=False).to(device)
        doc1_af         = autograd.Variable(torch.FloatTensor(doc1_af),   requires_grad=False).to(device)
        #########################
        doc1_sents_int_score    = self.snip_MLP_1(doc1_sents_embeds)
        #########################
        doc1_int_sent_scores_af = torch.cat((doc1_sents_int_score, doc1_sents_af), -1)
        #########################
        sents1_out              = self.snip_MLP_2(doc1_int_sent_scores_af).squeeze(-1)
        #########################
        max_feats_of_sents_1    = torch.max(sents1_out, 0)[0].unsqueeze(0)
        max_feats_of_sents_1_af = torch.cat((max_feats_of_sents_1, doc1_af), -1)
        #########################
        doc1_out                = self.doc_MLP(max_feats_of_sents_1_af)
        #########################
        return doc1_out, sents1_out
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, doc1_sents_af, doc2_sents_af, doc1_af, doc2_af):
        doc1_sents_af   = autograd.Variable(torch.FloatTensor(doc1_sents_af),   requires_grad=False).to(device)
        doc2_sents_af   = autograd.Variable(torch.FloatTensor(doc2_sents_af),   requires_grad=False).to(device)
        doc1_af         = autograd.Variable(torch.FloatTensor(doc1_af),   requires_grad=False).to(device)
        doc2_af         = autograd.Variable(torch.FloatTensor(doc2_af),   requires_grad=False).to(device)
        #########################
        doc1_sents_int_score    = self.snip_MLP_1(doc1_sents_embeds)
        doc2_sents_int_score    = self.snip_MLP_1(doc2_sents_embeds)
        #########################
        doc1_int_sent_scores_af = torch.cat((doc1_sents_int_score, doc1_sents_af), -1)
        doc2_int_sent_scores_af = torch.cat((doc2_sents_int_score, doc2_sents_af), -1)
        #########################
        sents1_out              = self.snip_MLP_2(doc1_int_sent_scores_af).squeeze(-1)
        sents2_out              = self.snip_MLP_2(doc2_int_sent_scores_af).squeeze(-1)
        #########################
        max_feats_of_sents_1    = torch.max(sents1_out, 0)[0].unsqueeze(0)
        max_feats_of_sents_1_af = torch.cat((max_feats_of_sents_1, doc1_af), -1)
        max_feats_of_sents_2    = torch.max(sents2_out, 0)[0].unsqueeze(0)
        max_feats_of_sents_2_af = torch.cat((max_feats_of_sents_2, doc2_af), -1)
        #########################
        doc1_out                = self.doc_MLP(max_feats_of_sents_1_af)
        doc2_out                = self.doc_MLP(max_feats_of_sents_2_af)
        #########################
        return doc1_out, sents1_out, doc2_out, sents2_out

#####################
# (model_class, tokenizer_class, pretrained_weights) = (BertModel, BertTokenizer, 'bert-base-uncased')
(model_class, tokenizer_class, pretrained_weights) = (RobertaModel, RobertaTokenizer, 'roberta-base')
#####################
bert_tokenizer  = tokenizer_class.from_pretrained(pretrained_weights)
# bert_model      = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
bert_model      = model_class.from_pretrained(pretrained_weights, output_hidden_states=False, output_attentions=False)
#####################
eval_path           = '/home/dpappas/bioasq_all/eval/run_eval.py'
retrieval_jar_path  = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'
odd                 = '/home/dpappas/'
#####################
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
dataloc             = '/home/dpappas/bioasq_all/bioasq7_data/'
#####################
bert_all_words_path = '/home/dpappas/bioasq_all/bert_all_words.pkl'
#####################
use_cuda            = torch.cuda.is_available()
device              = torch.device("cuda") if(use_cuda) else torch.device("cpu")
#####################
embedding_dim       = 768
b_size              = 6
max_epoch           = 10
#####################
lr                  = 1e-3
max_grad_norm       = 1.0
num_total_steps     = 1000
num_warmup_steps    = 100
warmup_proportion   = float(num_warmup_steps) / float(num_total_steps)  # 0.1
#####################

(dev_data, dev_docs, train_data, train_docs, idf, max_idf, bioasq6_data) = load_all_data(dataloc=dataloc, idf_pickle_path=idf_pickle_path)
hdlr = None
for run in range(0, 1):
    ######
    my_seed = run
    random.seed(my_seed)
    torch.manual_seed(my_seed)
    ######
    odir = 'bioasq7_pytorchTransformers_JROBERTA_2L_{}_run_{}/'.format(str(lr), run)
    # odir = 'bioasq7_pytorchTransformers_JBERT_2L_{}_run_{}/'.format(str(lr), run)
    odir = os.path.join(odd, odir)
    print(odir)
    if (not os.path.exists(odir)):
        os.makedirs(odir)
    ######
    logger, hdlr = init_the_logger(hdlr)
    print('random seed: {}'.format(my_seed))
    logger.info('random seed: {}'.format(my_seed))
    ######
    avgdl, mean, deviation = get_bm25_metrics(avgdl=21.2508, mean=0.5973, deviation=0.5926)
    print(avgdl, mean, deviation)
    ######
    print('Compiling model...')
    logger.info('Compiling model...')
    #####################
    model = JBert(768)
    print_params(model)
    #####################
    print_params(model)
    print_params(bert_model)
    #####################
    ### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
    #
    best_dev_map, test_map = None, None
    for epoch in range(max_epoch):
        train_one(epoch + 1, bioasq6_data, two_losses=True, use_sent_tokenizer=True)
        epoch_dev_map = get_one_map('dev', dev_data, dev_docs, use_sent_tokenizer=True)
        if (best_dev_map is None or epoch_dev_map >= best_dev_map):
            best_dev_map = epoch_dev_map
            save_checkpoint(epoch, model,      best_dev_map, optimizer, scheduler, filename=os.path.join(odir, 'best_checkpoint.pth.tar'))
            save_checkpoint(epoch, bert_model, best_dev_map, optimizer, scheduler, filename=os.path.join(odir, 'best_bert_checkpoint.pth.tar'))
        print('epoch:{:02d} epoch_dev_map:{:.4f} best_dev_map:{:.4f}'.format(epoch + 1, epoch_dev_map, best_dev_map))
        logger.info('epoch:{:02d} epoch_dev_map:{:.4f} best_dev_map:{:.4f}'.format(epoch + 1, epoch_dev_map, best_dev_map))

# CUDA_VISIBLE_DEVICES=1 python3.6 ttt2.py

'''
initializer_range   = 0.02
lr                  = 1e-3
max_grad_norm       = 1.0
num_total_steps     = 1000
num_warmup_steps    = 100
warmup_proportion   = float(num_warmup_steps) / float(num_total_steps)  # 0.1
### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
### and used like this:
for _ in range(1000):
    d1, s1, d2, s2  = model(doc1_sents_embeds, doc2_sents_embeds, doc1_sents_af, doc2_sents_af, doc1_af, doc2_af)
    loss            = my_hinge_loss(d1, d2)
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    print(d1, d2, loss, total_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
'''

'''
lr                  = 1e-3
optimizer = optim.Adam(list(model.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
for _ in range(10000):
    d1, s1, d2, s2  = model(doc1_sents_embeds, doc2_sents_embeds, doc1_sents_af, doc2_sents_af, doc1_af, doc2_af)
    loss            = my_hinge_loss(d1, d2)
    loss.backward()
    optimizer.step()
    print(d1, d2, loss)
'''

'''
sents1 = [
    'Look at my horse',
    'My horse is amazing',
    'Give it a lick',
    'Ooo, it tastes just like raisins',
]
sents2 = [
    'Have a stroke of its mane',
    'It turns into a plane',
    'And then it turns back again',
    'When you tug on its winkie',
    'Ooo, that’s dirty',
]
_, pooler_output    = encode_sents(sents1)
doc1_sents_embeds   = pooler_output
_, pooler_output    = encode_sents(sents2)
doc2_sents_embeds   = pooler_output
doc1_sents_af       = torch.zeros((len(sents1), 10))
doc2_sents_af       = torch.zeros((len(sents2), 10))
doc1_af             = torch.zeros((11))
doc2_af             = torch.zeros((11))
'''
