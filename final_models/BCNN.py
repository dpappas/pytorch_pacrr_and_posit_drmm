
import  os, json, time, random, re, nltk, pickle, logging, subprocess, math
import  torch
import  torch.nn.functional             as F
import  torch.nn                        as nn
import  numpy                           as np
import  torch.optim                     as optim
import  torch.autograd                  as autograd
from    tqdm                            import tqdm
from    pprint                          import pprint
from    gensim.models.keyedvectors      import KeyedVectors
from    nltk.tokenize                   import sent_tokenize
from    difflib                         import SequenceMatcher
from    keras.preprocessing.sequence    import pad_sequences
from    keras.utils                     import to_categorical
from    sklearn.metrics                 import roc_auc_score, f1_score

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
stopwords   = nltk.corpus.stopwords.words("english")

def get_bm25_metrics(avgdl=0., mean=0., deviation=0.):
    if(avgdl == 0):
        total_words = 0
        total_docs  = 0
        for dic in tqdm(train_docs):
            sents = sent_tokenize(train_docs[dic]['title']) + sent_tokenize(train_docs[dic]['abstractText'])
            for s in sents:
                total_words += len(tokenize(s))
                total_docs  += 1.
        avgdl = float(total_words) / float(total_docs)
    else:
        print('avgdl {} provided'.format(avgdl))
    #
    if(mean == 0 and deviation==0):
        BM25scores  = []
        k1, b       = 1.2, 0.75
        not_found   = 0
        for qid in tqdm(bioasq6_data):
            qtext           = bioasq6_data[qid]['body']
            all_retr_ids    = [link.split('/')[-1] for link in bioasq6_data[qid]['documents']]
            for dic in all_retr_ids:
                try:
                    sents   = sent_tokenize(train_docs[dic]['title']) + sent_tokenize(train_docs[dic]['abstractText'])
                    q_toks  = tokenize(qtext)
                    for sent in sents:
                        BM25score = similarity_score(q_toks, tokenize(sent), k1, b, idf, avgdl, False, 0, 0, max_idf)
                        BM25scores.append(BM25score)
                except KeyError:
                    not_found += 1
        #
        mean        = sum(BM25scores)/float(len(BM25scores))
        nominator   = 0
        for score in BM25scores:
            nominator += ((score - mean) ** 2)
        deviation   = math.sqrt((nominator) / float(len(BM25scores) - 1))
    else:
        print('mean {} provided'.format(mean))
        print('deviation {} provided'.format(deviation))
    return avgdl, mean, deviation

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
            score += rare_word * (
                    (tf(query_term, document) * (k1 + 1)) /
                    (
                            tf(query_term, document) +
                            k1 * (1 - b + b * (len(document) / avgdl))
                    )
            )
        else:
            score += idf_scores[query_term] * ((tf(query_term, document) * (k1 + 1)) / (tf(query_term, document) + k1 * (1 - b + b * (len(document) / avgdl))))
    if normalize:
        return ((score - mean)/deviation)
    else:
        return score

# Compute the average length from a collection of documents
def compute_avgdl(documents):
    total_words = 0
    for document in documents:
        total_words += len(document)
    avgdl = total_words / len(documents)
    return avgdl

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

def idf_val(w, idf, max_idf):
    try:
        return idf[w]
    except:
        return max_idf

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
          qwords_in_doc     += 1
          break
    if len(qwords) <= 0:
      qwords_in_doc_val = 0.0
    else:
      qwords_in_doc_val = (float(qwords_in_doc) / float(len(uwords(qwords))))
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

def snip_is_relevant(one_sent, gold_snips):
    return int(
        any(
            [
                (one_sent.encode('ascii','ignore')  in gold_snip.encode('ascii','ignore'))
                or
                (gold_snip.encode('ascii','ignore') in one_sent.encode('ascii','ignore'))
                for gold_snip in gold_snips
            ]
        )
    )

def tokenize(x):
  return bioclean(x)

def get_words(s, idf, max_idf):
    sl  = tokenize(s)
    sl  = [s for s in sl]
    sl2 = [s for s in sl if idf_val(s, idf, max_idf) >= 2.0]
    return sl, sl2

def GetScores(qtext, dtext, bm25, idf, max_idf):
    qwords, qw2 = get_words(qtext, idf, max_idf)
    dwords, dw2 = get_words(dtext, idf, max_idf)
    qd1         = query_doc_overlap(qwords, dwords, idf, max_idf)
    bm25        = [bm25]
    return qd1[0:3] + bm25

def get_embeds(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if(tok in wv):
            ret1.append(tok)
            ret2.append(wv[tok])
    return ret1, np.array(ret2, 'float64')

def get_embeds_use_unk(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if(tok in wv):
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

def prep_data(quest, the_doc, the_bm25, wv, good_snips, idf, max_idf):
    good_sents      = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
    ####
    quest_toks      = tokenize(quest)
    good_doc_af     = GetScores(quest, the_doc['title'] + the_doc['abstractText'], the_bm25, idf, max_idf)
    good_doc_af.append(len(good_sents) / 60.)
    ####
    good_sents_embeds, good_sents_escores, held_out_sents, good_sent_tags = [], [], [], []
    for good_text in good_sents:
        sent_toks                   = tokenize(good_text)
        good_tokens, good_embeds    = get_embeds(sent_toks, wv)
        # qwords_in_sent + qwords_bigrams_in_sent + idf_qwords_in_sent + doc_bm25
        good_escores                = GetScores(quest, good_text, the_bm25, idf, max_idf)[:-1]
        good_escores.append(len(sent_toks)/ 342.)
        if (len(good_embeds) > 0):
            tomi                = (set(sent_toks) & set(quest_toks))
            tomi_no_stop        = tomi - set(stopwords)
            BM25score           = similarity_score(quest_toks, sent_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
            tomi_no_stop_idfs   = [idf_val(w, idf, max_idf) for w in tomi_no_stop]
            tomi_idfs           = [idf_val(w, idf, max_idf) for w in tomi]
            quest_idfs          = [idf_val(w, idf, max_idf) for w in quest_toks]
            features            = [
                # already have it
                # already have it
                len(quest)              / 300.,
                len(good_text)          / 300.,
                len(tomi_no_stop)       / 100.,
                BM25score,
                sum(tomi_no_stop_idfs)  / 100.,
                sum(tomi_idfs)          / sum(quest_idfs),
            ]
            #
            good_sents_embeds.append(good_embeds)
            good_sents_escores.append(good_escores+features)
            held_out_sents.append(good_text)
            good_sent_tags.append(snip_is_relevant(' '.join(bioclean(good_text)), good_snips))
            #
    ####
    return {
        'sents_embeds'     : good_sents_embeds,
        'sents_escores'    : good_sents_escores,
        'sent_tags'        : good_sent_tags,
        'held_out_sents'   : held_out_sents,
    }

def get_snips(quest_id, gid, bioasq6_data):
    good_snips = []
    if('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            if(sn['document'].endswith(gid)):
                good_snips.extend(sent_tokenize(sn['text']))
    return good_snips

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
      doc_id    = data['queries'][i]['retrieved_documents'][j]['doc_id']
      year      = doc_text[doc_id]['publicationDate'].split('-')[0]
      ##########################
      # Skip 2017/2018 docs always. Skip 2016 docs for training.
      # Need to change for final model - 2017 should be a train year only.
      # Use only for testing.
      if year == '2017' or year == '2018' or (train and year == '2016'):
      #if year == '2018' or (train and year == '2017'):
        del data['queries'][i]['retrieved_documents'][j]
      else:
        j += 1
      ##########################
      if j == len(data['queries'][i]['retrieved_documents']):
        break
  return data

def get_the_mesh(the_doc):
    good_meshes = []
    if('meshHeadingsList' in the_doc):
        for t in the_doc['meshHeadingsList']:
            t = t.split(':', 1)
            t = t[1].strip()
            t = t.lower()
            good_meshes.append(t)
    elif('MeshHeadings' in the_doc):
        for mesh_head_set in the_doc['MeshHeadings']:
            for item in mesh_head_set:
                good_meshes.append(item['text'].strip().lower())
    if('Chemicals' in the_doc):
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
  for i in range(len(data['queries'])):
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

def load_all_data(dataloc, w2v_bin_path, idf_pickle_path):
    print('loading pickle data')
    #
    with open(dataloc+'BioASQ-trainingDataset6b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict( (q['id'], q) for q in bioasq6_data['questions'] )
    #
    with open(dataloc + 'bioasq_bm25_top100.test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.test.pkl', 'rb') as f:
        test_docs = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.dev.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.dev.pkl', 'rb') as f:
        dev_docs = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_docset_top100.train.pkl', 'rb') as f:
        train_docs = pickle.load(f)
    print('loading words')
    #
    train_data  = RemoveBadYears(train_data, train_docs, True)
    train_data  = RemoveTrainLargeYears(train_data, train_docs)
    dev_data    = RemoveBadYears(dev_data, dev_docs, False)
    test_data   = RemoveBadYears(test_data, test_docs, False)
    #
    words           = {}
    GetWords(train_data, train_docs, words)
    GetWords(dev_data,   dev_docs,   words)
    GetWords(test_data,  test_docs,  words)
    #
    print('loading idfs')
    idf, max_idf    = load_idfs(idf_pickle_path, words)
    print('loading w2v')
    wv              = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    wv              = dict([(word, wv[word]) for word in wv.vocab.keys() if(word in words)])
    return test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, wv, bioasq6_data

def train_data_step1(train_data):
    ret = []
    for dato in tqdm(train_data['queries']):
        quest       = dato['query_text']
        quest_id    = dato['query_id']
        bm25s       = {t['doc_id']: t['norm_bm25_score'] for t in dato[u'retrieved_documents']}
        ret_pmids   = [t[u'doc_id'] for t in dato[u'retrieved_documents']]
        good_pmids  = [t for t in ret_pmids if t in dato[u'relevant_documents']]
        bad_pmids   = [t for t in ret_pmids if t not in dato[u'relevant_documents']]
        if(len(bad_pmids)>0):
            for gid in good_pmids:
                bid = random.choice(bad_pmids)
                ret.append((quest, quest_id, gid, bid, bm25s[gid], bm25s[bid]))
    # print('')
    return ret

def train_data_step2(instances, docs, wv, bioasq6_data, idf, max_idf):
    for quest_text, quest_id, gid, bid, bm25s_gid, bm25s_bid in tqdm(instances):
    # for quest_text, quest_id, gid, bid, bm25s_gid, bm25s_bid in instances:
        good_snips                  = get_snips(quest_id, gid, bioasq6_data)
        good_snips                  = [' '.join(bioclean(sn)) for sn in good_snips]
        #
        datum                       = prep_data(quest_text, docs[gid], bm25s_gid, wv, good_snips, idf, max_idf)
        good_sents_embeds           = datum['sents_embeds']
        good_sents_escores          = datum['sents_escores']
        good_sent_tags              = datum['sent_tags']
        good_held_out_sents         = datum['held_out_sents']
        #
        datum                       = prep_data(quest_text, docs[bid], bm25s_bid, wv, [], idf, max_idf)
        bad_sents_embeds            = datum['sents_embeds']
        bad_sents_escores           = datum['sents_escores']
        bad_sent_tags               = [0] * len(datum['sent_tags'])
        bad_held_out_sents          = datum['held_out_sents']
        #
        quest_tokens, quest_embeds  = get_embeds(tokenize(quest_text), wv)
        q_idfs                      = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
        #
        if(sum(good_sent_tags)>0):
            yield {
                'good_sents_embeds'     : good_sents_embeds,
                'good_sents_escores'    : good_sents_escores,
                'good_sent_tags'        : good_sent_tags,
                'good_held_out_sents'   : good_held_out_sents,
                #
                'bad_sents_embeds'      : bad_sents_embeds,
                'bad_sents_escores'     : bad_sents_escores,
                'bad_sent_tags'         : bad_sent_tags,
                'bad_held_out_sents'    : bad_held_out_sents,
                #
                'quest_embeds'          : quest_embeds,
                'q_idfs'                : q_idfs,
            }

def compute_the_cost(costs, back_prop=True):
    cost_ = torch.stack(costs)
    cost_ = cost_.sum() / (1.0 * cost_.size(0))
    if(back_prop):
        cost_.backward()
        optimizer.step()
        optimizer.zero_grad()
    the_cost = cost_.cpu().item()
    return the_cost

def back_prop(batch_costs, epoch_costs):
    batch_cost      = sum(batch_costs) / float(len(batch_costs))
    batch_cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    batch_aver_cost = batch_cost.cpu().item()
    epoch_aver_cost = sum(epoch_costs) / float(len(epoch_costs))
    return batch_aver_cost, epoch_aver_cost

def save_checkpoint(epoch, model, max_dev_map, optimizer, filename='checkpoint.pth.tar'):
    '''
    :param state:       the stete of the pytorch mode
    :param filename:    the name of the file in which we will store the model.
    :return:            Nothing. It just saves the model.
    '''
    state = {
        'epoch':            epoch,
        'state_dict':       model.state_dict(),
        'best_valid_score': max_dev_map,
        'optimizer':        optimizer.state_dict(),
    }
    torch.save(state, filename)

def data_step1(data):
    ret = []
    for dato in tqdm(data['queries']):
        quest       = dato['query_text']
        quest_id    = dato['query_id']
        bm25s       = {t['doc_id']: t['norm_bm25_score'] for t in dato[u'retrieved_documents']}
        ret_pmids   = [t[u'doc_id'] for t in dato[u'retrieved_documents']]
        good_pmids  = [t for t in ret_pmids if t in dato[u'relevant_documents']]
        bad_pmids   = [t for t in ret_pmids if t not in dato[u'relevant_documents']]
        if(len(bad_pmids)>0):
            for gid in good_pmids:
                bid = random.choice(bad_pmids)
                ret.append((quest, quest_id, gid, bid, bm25s[gid], bm25s[bid]))
    # print('')
    return ret

def data_step2(instances, docs):
    for quest, quest_id, gid, bid, bm25s_gid, bm25s_bid in instances:
        quest_toks                              = tokenize(quest)
        quest_tokens, quest_embeds              = get_embeds(quest_toks, wv)
        q_idfs                                  = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
        #
        good_sents_title                        = sent_tokenize(docs[gid]['title'])
        good_sents_abs                          = sent_tokenize(docs[gid]['abstractText'])
        good_sents                              = good_sents_title + good_sents_abs
        good_snips                              = get_snips(quest_id, gid, bioasq6_data)
        good_snips                              = [' '.join(bioclean(sn)) for sn in good_snips]
        #
        good_sents_embeds, good_sents_escores, good_sent_tags = [], [], []
        for good_text in good_sents:
            sent_toks                           = tokenize(good_text)
            good_tokens, good_embeds            = get_embeds(sent_toks, wv)
            good_escores                        = GetScores(quest, good_text, bm25s_gid, idf, max_idf)[:-1]
            good_escores.append(len(sent_toks) / 342.)
            if(len(good_embeds)>0):
                tomi                            = (set(sent_toks) & set(quest_toks))
                tomi_no_stop                    = tomi - set(stopwords)
                BM25score                       = similarity_score(quest_toks, sent_toks, 1.2, 0.75, idf, avgdl, True, mean, deviation, max_idf)
                tomi_no_stop_idfs               = [idf_val(w, idf, max_idf) for w in tomi_no_stop]
                tomi_idfs                       = [idf_val(w, idf, max_idf) for w in tomi]
                quest_idfs                      = [idf_val(w, idf, max_idf) for w in quest_toks]
                features            = [
                    # already have it
                    # already have it
                    len(quest)              / 300.,
                    len(good_text)          / 300.,
                    len(tomi_no_stop)       / 100.,
                    BM25score,
                    sum(tomi_no_stop_idfs)  / 100.,
                    sum(tomi_idfs)          / sum(quest_idfs),
                ]
                good_sents_embeds.append(good_embeds)
                good_sents_escores.append(good_escores+features)
                tt                              = ' '.join(bioclean(good_text))
                good_sent_tags.append(snip_is_relevant(tt, good_snips))
        #
        if(sum(good_sent_tags)>0):
            yield (good_sents_embeds, quest_embeds, q_idfs, good_sents_escores, good_sent_tags)

def get_two_snip_losses(good_sent_tags, gs_emits_):
    gs_emits_       = gs_emits_.squeeze(-1)
    good_sent_tags  = torch.FloatTensor(good_sent_tags)
    if(use_cuda):
        good_sent_tags = good_sent_tags.cuda()
    sn_d1_l         = F.binary_cross_entropy(gs_emits_, good_sent_tags, reduction='elementwise_mean')
    return sn_d1_l

def train_one():
    model.train()
    epoch_labels, epoch_emits, epoch_costs  = [], [], []
    batch_labels, batch_emits, batch_costs  = [], [], []
    # batch_counter, instance_counter         = 0, 0
    start_time                              = time.time()
    train_instances                         = train_data_step1(train_data)
    random.shuffle(train_instances)
    for datum in train_data_step2(train_instances, train_docs, wv, bioasq6_data, idf, max_idf):
        gcost_, gemits_                     = model(
            sents_embeds                    = datum['good_sents_embeds'],
            question_embeds                 = datum['quest_embeds'],
            sents_gaf                       = datum['good_sents_escores'],
            sents_labels                    = datum['good_sent_tags']
        )
        bcost_, bemits_                     = model(
            sents_embeds                    = datum['bad_sents_embeds'],
            question_embeds                 = datum['quest_embeds'],
            sents_gaf                       = datum['bad_sents_escores'],
            sents_labels                    = datum['bad_sent_tags']
        )
        cost_                               = (gcost_ + bcost_) / 2.
        cost_                               = gcost_
        gemits_                             = gemits_.data.cpu().numpy().tolist()
        bemits_                             = bemits_.data.cpu().numpy().tolist()
        #
        batch_costs.append(cost_)
        epoch_costs.append(cost_)
        batch_labels.extend(datum['good_sent_tags'] + datum['bad_sent_tags'])
        epoch_labels.extend(datum['good_sent_tags'] + datum['bad_sent_tags'])
        batch_emits.extend(gemits_ + bemits_)
        epoch_emits.extend(gemits_ + bemits_)
        #
        # instance_counter += 1
        # if (instance_counter % b_size == 0):
        #     batch_counter += 1
        #     batch_auc = roc_auc_score(batch_labels, batch_emits)
        #     epoch_auc = roc_auc_score(epoch_labels, epoch_emits)
        #     batch_aver_cost, epoch_aver_cost = back_prop(batch_costs, epoch_costs)
        #     batch_labels, batch_emits, batch_costs = [], [], []
        #     elapsed_time = time.time() - start_time
        #     start_time                              = time.time()
        #     print('Epoch:{:02d} BatchCounter:{:03d} BatchAverCost:{:.4f} EpochAverCost:{:.4f} BatchAUC:{:.4f} EpochAUC:{:.4f} ElapsedTime:{:.4f}'.format(epoch+1, batch_counter, batch_aver_cost, epoch_aver_cost, batch_auc, epoch_auc, elapsed_time))
    #
    epoch_aver_cost                         = sum(epoch_costs) / float(len(epoch_costs))
    epoch_auc                               = roc_auc_score(epoch_labels, epoch_emits)
    elapsed_time                            = time.time() - start_time
    # start_time                              = time.time()
    print('Epoch:{:02d} EpochAverCost:{:.4f} EpochAUC:{:.4f} ElapsedTime:{:.4f}'.format(epoch+1, epoch_aver_cost, epoch_auc, elapsed_time))
    logger.info('Epoch:{:02d} EpochAverCost:{:.4f} EpochAUC:{:.4f} ElapsedTime:{:.4f}'.format(epoch+1, epoch_aver_cost, epoch_auc, elapsed_time))

def train_one_only_positive():
    model.train()
    epoch_labels, epoch_emits, epoch_costs  = [], [], []
    batch_labels, batch_emits, batch_costs  = [], [], []
    instance_counter, batch_counter         = 0, 0
    start_time                              = time.time()
    train_instances                         = train_data_step1(train_data)
    random.shuffle(train_instances)
    for datum in train_data_step2(train_instances, train_docs, wv, bioasq6_data, idf, max_idf):
        if (model_type in ['BCNN', 'ABCNN3']):
            gcost_, gemits_                     = model(
                sents_embeds                    = datum['good_sents_embeds'],
                question_embeds                 = datum['quest_embeds'],
                sents_gaf                       = datum['good_sents_escores'],
                sents_labels                    = datum['good_sent_tags']
            )
        else:
            gcost_, gemits_                     = model(
                doc1_sents_embeds               = datum['good_sents_embeds'],
                question_embeds                 = datum['quest_embeds'],
                q_idfs                          = datum['q_idfs'],
                sents_gaf                       = datum['good_sents_escores'],
                sents_labels                    = datum['good_sent_tags']
            )
        cost_                                   = gcost_
        gemits_                                 = gemits_.data.cpu().numpy().tolist()
        #
        batch_costs.append(cost_)
        epoch_costs.append(cost_)
        batch_labels.extend(datum['good_sent_tags'])
        epoch_labels.extend(datum['good_sent_tags'])
        batch_emits.extend(gemits_)
        epoch_emits.extend(gemits_)
        #
        instance_counter                            += 1
        if (instance_counter % b_size == 0):
            batch_counter                           += 1
            batch_aver_cost, epoch_aver_cost        = back_prop(batch_costs, epoch_costs)
            batch_labels, batch_emits, batch_costs  = [], [], []
    #
    epoch_aver_cost                         = sum(epoch_costs) / float(len(epoch_costs))
    epoch_auc                               = roc_auc_score(epoch_labels, epoch_emits)
    elapsed_time                            = time.time() - start_time
    print('Epoch:{:02d} EpochAverCost:{:.4f} EpochAUC:{:.4f} ElapsedTime:{:.4f}'.format(epoch+1, epoch_aver_cost, epoch_auc, elapsed_time))
    logger.info('Epoch:{:02d} EpochAverCost:{:.4f} EpochAUC:{:.4f} ElapsedTime:{:.4f}'.format(epoch+1, epoch_aver_cost, epoch_auc, elapsed_time))

def get_one_auc(prefix, data, docs):
    model.eval()
    #
    epoch_labels    = []
    epoch_emits     = []
    epoch_costs     = []
    instances       = data_step1(data)
    # random.shuffle(instances)
    for (good_sents_embeds, quest_embeds, q_idfs, good_sents_escores, good_sent_tags) in data_step2(instances, docs):
        if (model_type == 'BCNN'):
            _, gs_emits_       = model(
                sents_embeds        = good_sents_embeds,
                question_embeds     = quest_embeds,
                sents_gaf           = good_sents_escores,
                sents_labels        = good_sent_tags
            )
        else:
            _, gs_emits_                        = model(
                doc1_sents_embeds               = good_sents_embeds,
                question_embeds                 = quest_embeds,
                q_idfs                          = q_idfs,
                sents_gaf                       = good_sents_escores,
                sents_labels                    = good_sent_tags
            )
        #
        cost_                   = get_two_snip_losses(good_sent_tags, gs_emits_)
        gs_emits_               = gs_emits_.data.cpu().numpy().tolist()
        good_sent_tags          = good_sent_tags    + [0, 1]
        gs_emits_               = gs_emits_         + [0, 1]
        #
        epoch_costs.append(cost_)
        epoch_labels.extend(good_sent_tags)
        epoch_emits.extend(gs_emits_)
    #
    epoch_aver_cost             = sum(epoch_costs) / float(len(epoch_costs))
    epoch_aver_auc              = roc_auc_score(epoch_labels, epoch_emits)
    emit_tags                   = [1 if(e>=.5) else 0 for e in epoch_emits]
    epoch_aver_f1               = f1_score(epoch_labels, emit_tags)
    #
    print('{} Epoch:{} aver_epoch_cost: {} aver_epoch_auc: {} epoch_aver_f1: {}'.format(prefix, epoch+1, epoch_aver_cost, epoch_aver_auc, epoch_aver_f1))
    logger.info('{} Epoch:{} aver_epoch_cost: {} aver_epoch_auc: {} epoch_aver_f1: {}'.format(prefix, epoch+1, epoch_aver_cost, epoch_aver_auc, epoch_aver_f1))
    return epoch_aver_auc, epoch_aver_f1

def init_the_logger(hdlr):
    if not os.path.exists(odir):
        os.makedirs(odir)
    od          = odir.split('/')[-1] # 'sent_posit_drmm_MarginRankingLoss_0p001'
    logger      = logging.getLogger(od)
    if(hdlr is not None):
        logger.removeHandler(hdlr)
    hdlr        = logging.FileHandler(os.path.join(odir,'model.log'))
    formatter   = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr

def setup_random(run):
    np.random.seed(run)
    random.seed(run)
    torch.manual_seed(run)
    if(use_cuda):
        torch.cuda.manual_seed_all(run)

def setup_optim_model():
    if(model_type == 'ABCNN3'):
        model = ABCNN3(embedding_dim=embedding_dim)
    elif(model_type == 'BCNN'):
        model = BCNN(embedding_dim=embedding_dim, additional_feats=additional_feats, convolution_size=4)
    elif (model_type == 'BCNN_PDRMM'):
        model = BCNN_PDRMM(embedding_dim=embedding_dim)
    else:
        model = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim)
    params          = model.parameters()
    if(optim_type.lower() == 'sgd'):
        optimizer   = optim.SGD(params,     lr=lr, momentum=0.9)
    elif(optim_type.lower() == 'adam'):
        optimizer   = optim.Adam(params,    lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    else:
        optimizer   = optim.Adagrad(params, lr=lr, lr_decay=0.00001, weight_decay=0.0004, initial_accumulator_value=0)
    return model, optimizer

def load_model_from_checkpoint(model, resume_from_path):
    if os.path.isfile(resume_from_path):
        print("=> loading checkpoint '{}'".format(resume_from_path))
        checkpoint = torch.load(resume_from_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from_path, checkpoint['epoch']))

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim= 30):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        #
        self.embedding_dim              = embedding_dim
        self.k                          = 5
        # to create q weights
        self.trigram_conv_1             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_1  = torch.nn.LeakyReLU(negative_slope=0.1)
        self.trigram_conv_2             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_2  = torch.nn.LeakyReLU(negative_slope=0.1)
        # init_question_weight_module
        self.q_weights_mlp              = nn.Linear(self.embedding_dim+1, 1, bias=True)
        #
        self.convolution_size           = 3
        self.out_conv                   = nn.Conv1d(
            in_channels                 = 11,
            out_channels                = 2,
            kernel_size                 = self.convolution_size,
            padding                     = self.convolution_size-1,
            bias                        = True
        )
        self.init_mlps_for_pooled_attention()
        if(use_cuda):
            self.trigram_conv_1             = self.trigram_conv_1.cuda()
            self.trigram_conv_activation_1  = self.trigram_conv_activation_1.cuda()
            self.trigram_conv_2             = self.trigram_conv_2.cuda()
            self.trigram_conv_activation_2  = self.trigram_conv_activation_2.cuda()
            self.q_weights_mlp              = self.q_weights_mlp.cuda()
            self.out_conv                   = self.out_conv.cuda()
    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1      = nn.Linear(3 * 3, 8, bias=True)
        self.my_relu1           = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2      = nn.Linear(8, 1, bias=True)
        if(use_cuda):
            self.linear_per_q1  = self.linear_per_q1.cuda()
            self.my_relu1       = self.my_relu1.cuda()
            self.linear_per_q2  = self.linear_per_q2.cuda()
    def apply_context_convolution(self, the_input, the_filters, activation):
        conv_res        = the_filters(the_input.transpose(0,1).unsqueeze(0))
        if(activation is not None):
            conv_res    = activation(conv_res)
        pad             = the_filters.padding[0]
        ind_from        = int(np.floor(pad/2.0))
        ind_to          = ind_from + the_input.size(0)
        conv_res        = conv_res[:, :, ind_from:ind_to]
        conv_res        = conv_res.transpose(1, 2)
        conv_res        = conv_res + the_input
        return conv_res.squeeze(0)
    def my_cosine_sim(self, A, B):
        A           = A.unsqueeze(0)
        B           = B.unsqueeze(0)
        A_mag       = torch.norm(A, 2, dim=2)
        B_mag       = torch.norm(B, 2, dim=2)
        num         = torch.bmm(A, B.transpose(-1,-2))
        den         = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1,-2))
        dist_mat    = num / den
        return dist_mat
    def pooling_method(self, sim_matrix):
        sorted_res              = torch.sort(sim_matrix, -1)[0]                             # sort the input minimum to maximum
        k_max_pooled            = sorted_res[:,-self.k:]                                    # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)                        # average these k values
        the_maximum             = k_max_pooled[:, -1]                                       # select the maximum value of each instance
        the_average_over_all    = sorted_res.sum(-1)/float(sim_matrix.size(1))              # add average of all elements as long sentences might have more matches
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled, the_average_over_all], dim=-1)  # concatenate maximum value and average of k-max values
        return the_concatenation     # return the concatenation
    def get_output(self, input_list, weights):
        temp    = torch.cat(input_list, -1)
        lo      = self.linear_per_q1(temp)
        lo      = self.my_relu1(lo)
        lo      = self.linear_per_q2(lo)
        lo      = lo.squeeze(-1)
        lo      = lo * weights
        sr      = lo.sum(-1) / lo.size(-1)
        return sr
    def do_for_one_doc_cnn(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights):
        res = []
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            if(use_cuda):
                sent_embeds     = sent_embeds.cuda()
                gaf             = gaf.cuda()
            conv_res            = self.apply_context_convolution(sent_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
            conv_res            = self.apply_context_convolution(conv_res,      self.trigram_conv_2, self.trigram_conv_activation_2)
            #
            sim_insens          = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_oh              = (sim_insens > (1 - (1e-3))).float()
            sim_sens            = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled  = self.pooling_method(sim_insens)
            sensitive_pooled    = self.pooling_method(sim_sens)
            oh_pooled           = self.pooling_method(sim_oh)
            #
            sent_emit           = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats      = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        # self.out_conv
        res = torch.stack(res)
        # res = self.sent_out_layer(res)
        res = self.out_conv(res.transpose(-1,-2).unsqueeze(0))[:,:,1:res.size(0)+1]
        res = res.squeeze(0).transpose(-1,-2)
        return res
    def get_max_and_average_of_k_max(self, res, k):
        sorted_res              = torch.sort(res)[0]
        k_max_pooled            = sorted_res[-k:]
        average_k_max_pooled    = k_max_pooled.sum()/float(k)
        the_maximum             = k_max_pooled[-1]
        # print(the_maximum)
        # print(the_maximum.size())
        # print(average_k_max_pooled)
        # print(average_k_max_pooled.size())
        the_concatenation       = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation
    def get_max(self, res):
        return torch.max(res)
    def get_kmax(self, res):
        res     = torch.sort(res,0)[0]
        res     = res[-self.k2:].squeeze(-1)
        if(res.size()[0] < self.k2):
            res         = torch.cat([res, torch.zeros(self.k2 - res.size()[0])], -1)
        return res
    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res
    def get_mesh_rep(self, meshes_embeds, q_context):
        meshes_embeds   = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds   = torch.stack(meshes_embeds)
        sim_matrix      = self.my_cosine_sim(meshes_embeds, q_context).squeeze(0)
        max_sim         = torch.sort(sim_matrix, -1)[0][:, -1]
        output          = torch.mm(max_sim.unsqueeze(0), meshes_embeds)[0]
        return output
    def forward(self, doc1_sents_embeds, question_embeds, q_idfs, sents_gaf, sents_labels):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        sents_labels        = autograd.Variable(torch.LongTensor(sents_labels),        requires_grad=False)
        if(use_cuda):
            q_idfs          = q_idfs.cuda()
            question_embeds = question_embeds.cuda()
            sents_labels    = sents_labels.cuda()
        #
        q_context           = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context           = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        gs_emits            = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights)
        #
        mlp_out             = F.log_softmax(gs_emits, dim=-1)
        cost                = F.nll_loss(mlp_out, sents_labels, weight=None, reduction='elementwise_mean')
        #
        emit                = F.softmax(gs_emits, dim=-1)[:,1]
        return cost, emit

class BCNN_PDRMM(nn.Module):
    def __init__(self, embedding_dim= 30):
        super(BCNN_PDRMM, self).__init__()
        #
        self.embedding_dim              = embedding_dim
        self.k                          = 5
        # to create q weights
        self.trigram_conv_1             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_1  = torch.nn.LeakyReLU(negative_slope=0.1)
        self.trigram_conv_2             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_2  = torch.nn.LeakyReLU(negative_slope=0.1)
        # init_question_weight_module
        self.q_weights_mlp              = nn.Linear(self.embedding_dim+1, 1, bias=True)
        #
        self.convolution_size           = 3
        self.out_conv                   = nn.Conv1d(
            in_channels                 = 13,
            out_channels                = 2,
            kernel_size                 = self.convolution_size,
            padding                     = self.convolution_size-1,
            bias                        = True
        )
        self.init_mlps_for_pooled_attention()
        if(use_cuda):
            self.trigram_conv_1             = self.trigram_conv_1.cuda()
            self.trigram_conv_activation_1  = self.trigram_conv_activation_1.cuda()
            self.trigram_conv_2             = self.trigram_conv_2.cuda()
            self.trigram_conv_activation_2  = self.trigram_conv_activation_2.cuda()
            self.q_weights_mlp              = self.q_weights_mlp.cuda()
            self.out_conv                   = self.out_conv.cuda()
    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1      = nn.Linear(3 * 3, 8, bias=True)
        self.my_relu1           = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2      = nn.Linear(8, 1, bias=True)
        if(use_cuda):
            self.linear_per_q1  = self.linear_per_q1.cuda()
            self.my_relu1       = self.my_relu1.cuda()
            self.linear_per_q2  = self.linear_per_q2.cuda()
    def apply_context_convolution(self, the_input, the_filters, activation):
        conv_res        = the_filters(the_input.transpose(0,1).unsqueeze(0))
        if(activation is not None):
            conv_res    = activation(conv_res)
        pad             = the_filters.padding[0]
        ind_from        = int(np.floor(pad/2.0))
        ind_to          = ind_from + the_input.size(0)
        conv_res        = conv_res[:, :, ind_from:ind_to]
        conv_res        = conv_res.transpose(1, 2)
        conv_res        = conv_res + the_input
        return conv_res.squeeze(0)
    def my_cosine_sim(self, A, B):
        A           = A.unsqueeze(0)
        B           = B.unsqueeze(0)
        A_mag       = torch.norm(A, 2, dim=2)
        B_mag       = torch.norm(B, 2, dim=2)
        num         = torch.bmm(A, B.transpose(-1,-2))
        den         = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1,-2))
        dist_mat    = num / den
        return dist_mat
    def pooling_method(self, sim_matrix):
        sorted_res              = torch.sort(sim_matrix, -1)[0]                             # sort the input minimum to maximum
        k_max_pooled            = sorted_res[:,-self.k:]                                    # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)                        # average these k values
        the_maximum             = k_max_pooled[:, -1]                                       # select the maximum value of each instance
        the_average_over_all    = sorted_res.sum(-1)/float(sim_matrix.size(1))              # add average of all elements as long sentences might have more matches
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled, the_average_over_all], dim=-1)  # concatenate maximum value and average of k-max values
        return the_concatenation     # return the concatenation
    def get_output(self, input_list, weights):
        temp    = torch.cat(input_list, -1)
        lo      = self.linear_per_q1(temp)
        lo      = self.my_relu1(lo)
        lo      = self.linear_per_q2(lo)
        lo      = lo.squeeze(-1)
        lo      = lo * weights
        sr      = lo.sum(-1) / lo.size(-1)
        return sr
    def do_for_one_doc_cnn(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights, quest_global_pool, quest_cont_global_pool):
        res = []
        for i in range(len(doc_sents_embeds)):
            sent_embeds             = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                     = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            if(use_cuda):
                sent_embeds         = sent_embeds.cuda()
                gaf                 = gaf.cuda()
            #
            conv_res                = self.apply_context_convolution(sent_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
            conv_res                = self.apply_context_convolution(conv_res,      self.trigram_conv_2, self.trigram_conv_activation_2)
            #
            sent_global_pool        = self.glob_average_pool(sent_embeds)
            glob_aver_sim           = self.my_cosine_sim(quest_global_pool, sent_global_pool).squeeze(0)
            sent_cont_global_pool   = self.glob_average_pool(conv_res)
            glob_cont_aver_sim      = self.my_cosine_sim(quest_cont_global_pool, sent_cont_global_pool).squeeze(0)
            #
            sim_insens              = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_oh                  = (sim_insens > (1 - (1e-3))).float()
            sim_sens                = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled      = self.pooling_method(sim_insens)
            sensitive_pooled        = self.pooling_method(sim_sens)
            oh_pooled               = self.pooling_method(sim_oh)
            #
            sent_emit               = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats          = torch.cat([gaf, sent_emit.unsqueeze(-1), glob_aver_sim.squeeze(-1), glob_cont_aver_sim.squeeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        # res = self.sent_out_layer(res)
        res = self.out_conv(res.transpose(-1,-2).unsqueeze(0))[:,:,1:res.size(0)+1]
        res = res.squeeze(0).transpose(-1,-2)
        return res
    def get_max_and_average_of_k_max(self, res, k):
        sorted_res              = torch.sort(res)[0]
        k_max_pooled            = sorted_res[-k:]
        average_k_max_pooled    = k_max_pooled.sum()/float(k)
        the_maximum             = k_max_pooled[-1]
        # print(the_maximum)
        # print(the_maximum.size())
        # print(average_k_max_pooled)
        # print(average_k_max_pooled.size())
        the_concatenation       = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation
    def get_max(self, res):
        return torch.max(res)
    def get_kmax(self, res):
        res     = torch.sort(res,0)[0]
        res     = res[-self.k2:].squeeze(-1)
        if(res.size()[0] < self.k2):
            res         = torch.cat([res, torch.zeros(self.k2 - res.size()[0])], -1)
        return res
    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res
    def get_mesh_rep(self, meshes_embeds, q_context):
        meshes_embeds   = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds   = torch.stack(meshes_embeds)
        sim_matrix      = self.my_cosine_sim(meshes_embeds, q_context).squeeze(0)
        max_sim         = torch.sort(sim_matrix, -1)[0][:, -1]
        output          = torch.mm(max_sim.unsqueeze(0), meshes_embeds)[0]
        return output
    def glob_average_pool(self, the_input):
        the_input               = the_input.unsqueeze(0).transpose(-1,-2)
        the_input_global_pool   = F.avg_pool1d(the_input, the_input.size(-1), stride=None)
        return the_input_global_pool.squeeze(-1)
    def forward(self, doc1_sents_embeds, question_embeds, q_idfs, sents_gaf, sents_labels):
        q_idfs                  = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds         = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        sents_labels            = autograd.Variable(torch.LongTensor(sents_labels),        requires_grad=False)
        if(use_cuda):
            q_idfs              = q_idfs.cuda()
            question_embeds     = question_embeds.cuda()
            sents_labels        = sents_labels.cuda()
        #
        q_context               = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context               = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        quest_global_pool       = self.glob_average_pool(question_embeds)
        quest_cont_global_pool  = self.glob_average_pool(q_context)
        #
        q_weights               = torch.cat([q_context, q_idfs], -1)
        q_weights               = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights               = F.softmax(q_weights, dim=-1)
        #
        gs_emits                = self.do_for_one_doc_cnn(
            doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights, quest_global_pool, quest_cont_global_pool
        )
        #
        mlp_out                 = F.log_softmax(gs_emits, dim=-1)
        cost                    = F.nll_loss(mlp_out, sents_labels, weight=None, reduction='elementwise_mean')
        #
        emit                    = F.softmax(gs_emits, dim=-1)[:,1]
        return cost, emit

class BCNN(nn.Module):
    def __init__(self, embedding_dim=30, additional_feats=10, convolution_size=4):
        super(BCNN, self).__init__()
        self.additional_feats   = additional_feats
        self.convolution_size   = convolution_size
        self.embedding_dim      = embedding_dim
        self.conv1              = nn.Conv1d(
            in_channels         = self.embedding_dim,
            out_channels        = self.embedding_dim,
            kernel_size         = self.convolution_size,
            padding             = self.convolution_size-1,
            bias                = True
        )
        self.conv2              = nn.Conv1d(
            in_channels         = self.embedding_dim,
            out_channels        = self.embedding_dim,
            kernel_size         = self.convolution_size,
            padding             = self.convolution_size-1,
            bias                = True
        )
        self.linear_out         = nn.Linear(self.additional_feats+3, 2, bias=True)
        self.conv1_activ        = torch.nn.Tanh()
        if(use_cuda):
            self.linear_out     = self.linear_out.cuda()
            self.conv1_activ    = self.conv1_activ.cuda()
            self.conv1          = self.conv1.cuda()
            self.conv2          = self.conv2.cuda()
            self.conv1_activ    = self.conv1_activ.cuda()
    def my_cosine_sim(self, A, B):
        A_mag = torch.norm(A, 2, dim=2)
        B_mag = torch.norm(B, 2, dim=2)
        num = torch.bmm(A, B.transpose(-1, -2))
        den = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
        dist_mat = num / den
        return dist_mat
    def apply_one_conv(self, batch_x1, batch_x2, the_conv):
        batch_x1_conv       = the_conv(batch_x1)
        batch_x2_conv       = the_conv(batch_x2)
        #
        x1_window_pool      = F.avg_pool1d(batch_x1_conv, self.convolution_size, stride=1)
        x2_window_pool      = F.avg_pool1d(batch_x2_conv, self.convolution_size, stride=1)
        #
        x1_global_pool      = F.avg_pool1d(batch_x1_conv, batch_x1_conv.size(-1), stride=None)
        x2_global_pool      = F.avg_pool1d(batch_x2_conv, batch_x2_conv.size(-1), stride=None)
        #
        sim                 = self.my_cosine_sim(x1_global_pool.transpose(1,2), x2_global_pool.transpose(1,2))
        sim                 = sim.squeeze(-1).squeeze(-1)
        return x1_window_pool, x2_window_pool, x1_global_pool, x2_global_pool, sim
    def forward(self, sents_embeds, question_embeds, sents_gaf, sents_labels):
        sents_labels        = autograd.Variable(torch.LongTensor(sents_labels),     requires_grad=False)
        sents_gaf           = autograd.Variable(torch.FloatTensor(sents_gaf),       requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds), requires_grad=False).unsqueeze(0).transpose(-1, -2)
        if(use_cuda):
            sents_labels    = sents_labels.cuda()
            sents_gaf       = sents_gaf.cuda()
            question_embeds = question_embeds.cuda()
        quest_global_pool   = F.avg_pool1d(question_embeds, question_embeds.size(-1), stride=None)
        #
        mlp_in = []
        for i in range(len(sents_embeds)):
            sent_embed          = autograd.Variable(torch.FloatTensor(sents_embeds[i]), requires_grad=False).unsqueeze(0).transpose(-1,-2)
            if (use_cuda):
                sent_embed      = sent_embed.cuda()
            sent_global_pool    = F.avg_pool1d(sent_embed, sent_embed.size(-1), stride=None)
            sim1                = self.my_cosine_sim(
                quest_global_pool.transpose(-1, -2), sent_global_pool.transpose(-1, -2)
            ).squeeze(-1).squeeze(-1)
            (
                quest_window_pool, sent_window_pool, quest_global_pool, sent_global_pool, sim2
            ) = self.apply_one_conv(question_embeds, sent_embed, self.conv1)
            (
                quest_window_pool, sent_window_pool, quest_global_pool, sent_global_pool, sim3
            ) = self.apply_one_conv(quest_window_pool, sent_window_pool, self.conv2)
            mlp_in.append(torch.cat([sim1, sim2, sim3, sents_gaf[i]], dim=-1))
        #
        mlp_in              = torch.stack(mlp_in, dim=0)
        mlp_out             = self.linear_out(mlp_in)
        #
        mlp_out             = F.log_softmax(mlp_out, dim=-1)
        cost                = F.nll_loss(mlp_out, sents_labels, weight=None, reduction='elementwise_mean')
        #
        emit                = F.softmax(mlp_out, dim=-1)[:,1]
        return cost, emit

class ABCNN3(nn.Module):
    def __init__(self, embedding_dim=30, convolution_size=4):
        super(ABCNN3, self).__init__()
        self.embedding_dim      = embedding_dim
        self.conv1              = nn.Conv1d(
            in_channels         = self.embedding_dim,
            out_channels        = self.embedding_dim,
            kernel_size         = 4,
            padding             = 3,
            bias                = True
        )
        self.conv2              = nn.Conv1d(
            in_channels         = self.embedding_dim,
            out_channels        = self.embedding_dim,
            kernel_size         = 4,
            padding             = 3,
            bias                = True
        )
        self.linear_out         = nn.Linear(12, 2, bias=True)
        self.conv1_activ        = torch.nn.Tanh()
        if(use_cuda):
            self.linear_out     = self.linear_out.cuda()
            self.conv1_activ    = self.conv1_activ.cuda()
            self.conv1          = self.conv1.cuda()
            self.conv2          = self.conv2.cuda()
            self.conv1_activ    = self.conv1_activ.cuda()
    def my_cosine_sim(self, A, B):
        A_mag = torch.norm(A, 2, dim=2)
        B_mag = torch.norm(B, 2, dim=2)
        num = torch.bmm(A, B.transpose(-1, -2))
        den = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
        dist_mat = num / den
        return dist_mat
    def make_attention_mat(self, x1, x2):
        ret = []
        for i in range(x2.size(-1)):
            t   = x2[:,:,i].unsqueeze(-1).expand_as(x1)
            dif = torch.pow(x1 - t, 2)
            dif = torch.sum(dif, dim=1)
            ret.append(dif)
        ret = torch.stack(ret).permute(1,2,0)
        ret = torch.sqrt(ret)
        return ret
    def apply_one_conv(self, batch_x1, batch_x2, the_conv):
        print(batch_x1.size())
        print(batch_x2.size())
        att_mat             = self.make_attention_mat(batch_x1, batch_x2)
        print(att_mat.size())
        #
        batch_x1_conv       = the_conv(batch_x1)
        batch_x2_conv       = the_conv(batch_x2)
        print(batch_x1_conv.size())
        print(batch_x2_conv.size())
        exit()
        #
        x1_window_pool      = F.avg_pool1d(batch_x1_conv, self.convolution_size, stride=1)
        x2_window_pool      = F.avg_pool1d(batch_x2_conv, self.convolution_size, stride=1)
        #
        x1_global_pool      = F.avg_pool1d(batch_x1_conv, batch_x1_conv.size(-1), stride=None)
        x2_global_pool      = F.avg_pool1d(batch_x2_conv, batch_x2_conv.size(-1), stride=None)
        #
        sim                 = self.my_cosine_sim(x1_global_pool.transpose(1,2), x2_global_pool.transpose(1,2))
        sim                 = sim.squeeze(-1).squeeze(-1)
        return x1_window_pool, x2_window_pool, x1_global_pool, x2_global_pool, sim
    def forward(self, sents_embeds, question_embeds, sents_gaf, sents_labels):
        sents_labels        = autograd.Variable(torch.LongTensor(sents_labels),     requires_grad=False)
        sents_gaf           = autograd.Variable(torch.FloatTensor(sents_gaf),       requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds), requires_grad=False).unsqueeze(0).transpose(-1, -2)
        if(use_cuda):
            sents_labels    = sents_labels.cuda()
            sents_gaf       = sents_gaf.cuda()
            question_embeds = question_embeds.cuda()
        quest_global_pool   = F.avg_pool1d(question_embeds, question_embeds.size(-1), stride=None)
        #
        mlp_in = []
        for i in range(len(sents_embeds)):
            sent_embed          = autograd.Variable(torch.FloatTensor(sents_embeds[i]), requires_grad=False).unsqueeze(0).transpose(-1,-2)
            if (use_cuda):
                sent_embed      = sent_embed.cuda()
            sent_global_pool    = F.avg_pool1d(sent_embed, sent_embed.size(-1), stride=None)
            sim1                = self.my_cosine_sim(
                quest_global_pool.transpose(-1, -2), sent_global_pool.transpose(-1, -2)
            ).squeeze(-1).squeeze(-1)
            (
                quest_window_pool, sent_window_pool, quest_global_pool, sent_global_pool, sim2
            ) = self.apply_one_conv(question_embeds, sent_embed, self.conv1)
            (
                quest_window_pool, sent_window_pool, quest_global_pool, sent_global_pool, sim3
            ) = self.apply_one_conv(quest_window_pool, sent_window_pool, self.conv2)
            mlp_in.append(torch.cat([sim1, sim2, sim3, sents_gaf[i]], dim=-1))
        #
        mlp_in              = torch.stack(mlp_in, dim=0)
        mlp_out             = self.linear_out(mlp_in)
        #
        mlp_out             = F.log_softmax(mlp_out, dim=-1)
        cost                = F.nll_loss(mlp_out, sents_labels, weight=None, reduction='elementwise_mean')
        #
        emit                = F.softmax(mlp_out, dim=-1)[:,1]
        return cost, emit

# laptop
w2v_bin_path        = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/for_ryan/fordp/idf.pkl'
dataloc             = '/home/dpappas/for_ryan/'
eval_path           = '/home/dpappas/for_ryan/eval/run_eval.py'
retrieval_jar_path  = '/home/dpappas/NetBeansProjects/my_bioasq_eval_2/dist/my_bioasq_eval_2.jar'
use_cuda            = True
odd                 = '/home/dpappas/'
get_embeds          = get_embeds_use_unk

# # atlas , cslab243
# w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
# dataloc             = '/home/dpappas/bioasq_all/bioasq_data/'
# eval_path           = '/home/dpappas/bioasq_all/eval/run_eval.py'
# retrieval_jar_path  = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'
# use_cuda            = True
# odd                 = '/home/dpappas/'
# get_embeds          = get_embeds_use_unk
# # get_embeds          = get_embeds_use_only_unk

embedding_dim       = 30
additional_feats    = 10
b_size              = 32

(test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, wv, bioasq6_data) = load_all_data(dataloc=dataloc, w2v_bin_path=w2v_bin_path, idf_pickle_path=idf_pickle_path)

avgdl, mean, deviation = get_bm25_metrics(avgdl=21.2508, mean=0.5973, deviation=0.5926)
print(avgdl, mean, deviation)

# for model_type in ['BCNN_PDRMM', 'BCNN', 'PDRMM']:
# for model_type in ['BCNN_PDRMM']:

# model_type          = 'BCNN_PDRMM'
# model_type          = 'BCNN'
# model_type          = 'PDRMM'
model_type          = 'ABCNN3'
optim_type          = 'ADAM'
lr                  = 0.001
model, optimizer    = setup_optim_model()

# resume_from_path    = '/home/dpappas/PDRMM_ADAM_001_run_0/best_checkpoint.pth.tar'
# load_model_from_checkpoint(model, resume_from_path)
# print(model.out_conv.weight[0].sum())

hdlr                = None
for run in range(1):
    setup_random(run)
    #
    odir            = '/home/dpappas/{}_{}_{}_run_{}/'.format(model_type, optim_type, str(lr).replace('.',''), run)
    logger, hdlr    = init_the_logger(hdlr)
    if (not os.path.exists(odir)):
        os.makedirs(odir)
    print(odir)
    logger.info(odir)
    #
    best_dev_auc, test_auc, best_dev_epoch, best_dev_f1 = None, None, None, None
    for epoch in range(10):
        logger.info('Training...')
        train_one_only_positive()
        logger.info('Validating...')
        epoch_dev_auc, epoch_dev_f1 = get_one_auc('dev', dev_data, dev_docs)
        if(best_dev_auc is None or epoch_dev_auc>=best_dev_auc):
            best_dev_epoch  = epoch+1
            best_dev_auc    = epoch_dev_auc
            best_dev_f1     = epoch_dev_f1
            logger.info('Testing...')
            test_auc, test_f1 = get_one_auc('test', test_data, test_docs)
            save_checkpoint(epoch, model, best_dev_auc, optimizer, filename=odir+'best_checkpoint.pth.tar')
        print(
            'epoch:{} '
            'epoch_dev_auc:{} '
            'best_dev_auc:{} '
            'test_auc:{} '
            'best_dev_epoch:{} '
            'best_dev_f1:{} '
            'test_f1:{} '
            '\n'.format(
                epoch + 1, epoch_dev_auc, best_dev_auc, test_auc, best_dev_epoch, epoch_dev_f1, test_f1
            )

        )
        logger.info(
            'epoch:{} '
            'epoch_dev_auc:{} '
            'best_dev_auc:{} '
            'test_auc:{} '
            'best_dev_epoch:{} '
            'best_dev_f1:{} '
            'test_f1:{} '
            '\n'.format(
                epoch + 1, epoch_dev_auc, best_dev_auc, test_auc, best_dev_epoch, epoch_dev_f1, test_f1
            )

        )





