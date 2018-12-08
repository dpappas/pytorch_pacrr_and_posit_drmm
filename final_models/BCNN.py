
import  os, json, time, random, re, nltk, pickle, logging, subprocess
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
from    sklearn.metrics                 import roc_auc_score

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
stopwords   = nltk.corpus.stopwords.words("english")

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
    # print one_sent
    # pprint(gold_snips)
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
    good_sents  = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
    ####
    quest_toks  = tokenize(quest)
    good_doc_af = GetScores(quest, the_doc['title'] + the_doc['abstractText'], the_bm25, idf, max_idf)
    good_doc_af.append(len(good_sents) / 60.)
    ####
    good_sents_embeds, good_sents_escores, held_out_sents, good_sent_tags = [], [], [], []
    for good_text in good_sents:
        sent_toks                   = tokenize(good_text)
        good_tokens, good_embeds    = get_embeds(sent_toks, wv)
        good_escores                = GetScores(quest, good_text, the_bm25, idf, max_idf)[:-1]
        good_escores.append(len(sent_toks)/ 342.)
        if (len(good_embeds) > 0):
            tomi            = (set(sent_toks) & set(quest_toks))
            tomi_no_stop    = tomi - set(stopwords)
            features    = [
                len(quest),
                len(good_text),
                len(tomi_no_stop),
                sum([idf_val(w, idf, max_idf) for w in tomi_no_stop]),
                sum([idf_val(w, idf, max_idf) for w in tomi]) / sum([idf_val(w, idf, max_idf) for w in quest_toks]),
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
    print('')
    return ret

def train_data_step2(instances, docs, wv, bioasq6_data, idf, max_idf):
    # for quest_text, quest_id, gid, bid, bm25s_gid, bm25s_bid in tqdm(instances):
    for quest_text, quest_id, gid, bid, bm25s_gid, bm25s_bid in instances:
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

def dummy_test():
    model.train()
    bx1         = np.random.randn(b_size, max_len, embedding_dim)
    bx2         = np.random.randn(b_size, max_len, embedding_dim)
    by          = np.random.randint(2, size=b_size)
    bf          = np.random.randn(b_size, 8)
    for i in range(500):
        cost_ = model(
            batch_x1        = bx1,
            batch_x2        = bx2,
            batch_y         = by,
            batch_features  = bf
        )
        cost_.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(cost_)

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

class BCNN(nn.Module):
    def __init__(self, embedding_dim=30, additional_feats=8, convolution_size=4):
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
        # print(label, mlp_out)
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

embedding_dim       = 30
additional_feats    = 9
b_size              = 32
max_len             = 40
lr                  = 0.08

model = BCNN(
    embedding_dim       = embedding_dim,
    additional_feats    = additional_feats,
    convolution_size    = 4
)

params      = model.parameters()
# optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0004)
optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer   = optim.Adagrad(params, lr=lr, lr_decay=0.00001, weight_decay=0.0004, initial_accumulator_value=0)

(
    test_data, test_docs, dev_data, dev_docs, train_data, train_docs, idf, max_idf, wv, bioasq6_data
) = load_all_data(dataloc=dataloc, w2v_bin_path=w2v_bin_path, idf_pickle_path=idf_pickle_path)

train_instances = train_data_step1(train_data)

for epoch in range(10):
    epoch_labels, epoch_emits, epoch_costs = [], [], []
    batch_labels, batch_emits, batch_costs = [], [], []
    batch_counter   = 0
    epoch           += 1
    start_time      = time.time()
    for datum in train_data_step2(train_instances, train_docs, wv, bioasq6_data, idf, max_idf):
        gcost_, gemits_     = model(
            sents_embeds    = datum['good_sents_embeds'],
            question_embeds = datum['quest_embeds'],
            sents_gaf       = datum['good_sents_escores'],
            sents_labels    = datum['good_sent_tags']
        )
        bcost_, bemits_     = model(
            sents_embeds    = datum['bad_sents_embeds'],
            question_embeds = datum['quest_embeds'],
            sents_gaf       = datum['bad_sents_escores'],
            sents_labels    = datum['bad_sent_tags']
        )
        cost_               = (gcost_ + bcost_) / 2.
        gemits_             = gemits_.data.cpu().numpy().tolist()
        bemits_             = bemits_.data.cpu().numpy().tolist()
        #
        batch_costs.append(cost_)
        epoch_costs.append(cost_)
        batch_labels.extend(datum['good_sent_tags'] + datum['bad_sent_tags'])
        epoch_labels.extend(datum['good_sent_tags'] + datum['bad_sent_tags'])
        batch_emits.extend(gemits_+bemits_)
        epoch_emits.extend(gemits_+bemits_)
        #
        if(batch_counter % b_size == 0):
            batch_counter                           += 1
            batch_auc                               = roc_auc_score(batch_labels, batch_emits)
            epoch_auc                               = roc_auc_score(epoch_labels, epoch_emits)
            batch_aver_cost, epoch_aver_cost        = back_prop(batch_costs, epoch_costs)
            batch_labels, batch_emits, batch_costs  = [], [], []
            elapsed_time                            = time.time() - start_time
            start_time                              = time.time()
            print('Epoch:{:02d} BatchCounter:{:03d} BatchAverCost:{:.4f} EpochAverCost:{:.4f} BatchAUC:{:.4f} EpochAUC:{:.4f} ElapsedTime:{:.4f}'.format(epoch, batch_counter, batch_aver_cost, epoch_aver_cost, batch_auc, epoch_auc, elapsed_time))
    #
    epoch_aver_cost = sum(epoch_costs) / float(len(epoch_costs))
    epoch_auc       = roc_auc_score(epoch_labels, epoch_emits)
    elapsed_time    = time.time() - start_time
    start_time      = time.time()
    print('Epoch:{:02d} EpochAverCost:{:.4f} EpochAUC:{:.4f} ElapsedTime:{:.4f}'.format(epoch, epoch_aver_cost, epoch_auc, elapsed_time))




