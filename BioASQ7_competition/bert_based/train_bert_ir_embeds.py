
import json, torch, re, pickle, random, os
from pytorch_transformers import BertModel, BertTokenizer
import  torch.nn as nn
import  torch.optim             as optim
import  torch.nn.functional     as F
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

######################################################################

def load_all_data(dataloc):
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
    return dev_data, dev_docs, train_data, train_docs, bioasq6_data

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
    return ret

######################################################################

use_cuda    = torch.cuda.is_available()
device      = torch.device("cuda") if(use_cuda) else torch.device("cpu")

scibert_dir     = '/home/dpappas/scibert_scivocab_uncased'
bert_tokenizer  = BertTokenizer.from_pretrained(scibert_dir)
bert_model      = BertModel.from_pretrained(scibert_dir,  output_hidden_states=False, output_attentions=False).to(device)

for param in bert_model.parameters():
    param.requires_grad = False

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

def yield_batches(data, batch_size):
    instances       = train_data_step1(data)
    random.shuffle(instances)
    batch_good      = []
    batch_bad       = []
    batch_quests    = []
    for quest_text, quest_id, gid, bid, bm25s_gid, bm25s_bid in instances:
        good_doc    = train_docs[gid]['title'] + '--------------------' + train_docs[gid]['abstractText']
        bad_doc     = train_docs[bid]['title'] + '--------------------' + train_docs[bid]['abstractText']
        #################################################################################################
        batch_quests.append(quest_text)
        batch_good.append(good_doc)
        batch_bad.append(bad_doc)
        #################################################################################################
        if(len(batch_bad) == batch_size):
            yield batch_quests, batch_good, batch_bad
            batch_good      = []
            batch_bad       = []
            batch_quests    = []
    yield batch_quests, batch_good, batch_bad

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""
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

def embed_the_sents(sents):
    eval_examples = []
    for sent in sents:
        eval_examples.append(InputExample(guid='example_dato_1', text_a=' '.join(sent), text_b=None, label='1'))
    eval_features   = convert_examples_to_features(eval_examples, 256, bert_tokenizer)
    input_ids   = torch.tensor([ef.input_ids for ef in eval_features], dtype=torch.long).to(device)
    with torch.no_grad():
        timestep_outputs, sent_embeds = bert_model(input_ids)
    return timestep_outputs, sent_embeds

class DocEmbeder(nn.Module):
    def __init__(self, embedding_dim=100, input_dim=768, hidde_dim=256):
        super(DocEmbeder, self).__init__()
        self.embedding_dim  = 100
        self.hidde_size     = 256
        self.layer1         = nn.Linear(input_dim, hidde_dim, bias=True).to(device)
        self.layer2         = nn.Linear(hidde_dim, embedding_dim, bias=True).to(device)
        self.cos            = nn.CosineSimilarity(dim=1, eps=1e-6)
    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta))
        return loss_q_pos
    def forward(self, doc_vectors):
        l1 = F.leaky_relu(self.layer1(doc_vectors))
        l2 = F.leaky_relu(self.layer2(l1))
        return l2

TRAIN_BATCH_SIZE            = 24
LEARNING_RATE               = 2e-5
NUM_TRAIN_EPOCHS            = 1
RANDOM_SEED                 = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION           = 0.1

dataloc         = '/home/dpappas/bioasq_all/bioasq7_data/'
(dev_data, dev_docs, train_data, train_docs, bioasq6_data) = load_all_data(dataloc=dataloc)

batch_size      = 64

model       = DocEmbeder(embedding_dim=100, input_dim=768, hidde_dim=256).to(device)

optimizer = BertAdam(
    model.parameters(),
    lr          = LEARNING_RATE,
    warmup      = WARMUP_PROPORTION,
    t_total     = 10000
)

def train_one():
    model.train()
    all_losses = []
    pbar = tqdm(list(yield_batches(train_data, batch_size)))
    for batch_quests, batch_good, batch_bad in pbar:
        _, embeds_docs_neg      = embed_the_sents(batch_bad)
        _, embeds_docs_pos      = embed_the_sents(batch_good)
        _, embeds_quests        = embed_the_sents(batch_quests)
        #################################################################
        pos_vec_docs    = model(embeds_docs_pos)
        neg_vec_docs    = model(embeds_docs_neg)
        vec_quests      = model(embeds_quests)
        #################################################################
        pos_sim         = model.cos(pos_vec_docs, vec_quests)
        neg_sim         = model.cos(neg_vec_docs, vec_quests)
        loss            = model.my_hinge_loss(pos_sim, neg_sim, margin=0.4)
        all_losses.append(loss.cpu().item())
        #################################################################
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #################################################################
        pbar.set_description('{}'.format(sum(all_losses) / float(len(all_losses))))
        #################################################################
    return sum(all_losses) / float(len(all_losses))

def eval_one():
    model.eval()
    all_losses = []
    pbar = tqdm(list(yield_batches(dev_data, batch_size)))
    for batch_quests, batch_good, batch_bad in pbar:
        _, embeds_docs_neg      = embed_the_sents(batch_bad)
        _, embeds_docs_pos      = embed_the_sents(batch_good)
        _, embeds_quests        = embed_the_sents(batch_quests)
        #################################################################
        pos_vec_docs    = model(embeds_docs_pos)
        neg_vec_docs    = model(embeds_docs_neg)
        vec_quests      = model(embeds_quests)
        #################################################################
        pos_sim         = model.cos(pos_vec_docs, vec_quests)
        neg_sim         = model.cos(neg_vec_docs, vec_quests)
        loss            = model.my_hinge_loss(pos_sim, neg_sim, margin=0.4)
        all_losses.append(loss.cpu().item())
        #################################################################
        pbar.set_description('{}'.format(sum(all_losses) / float(len(all_losses))))
        #################################################################
    return sum(all_losses) / float(len(all_losses))

def save_checkpoint(epoch, model, max_dev_map, optimizer, filename='checkpoint.pth.tar'):
    state = {
        'epoch'             : epoch,
        'model_state_dict'  : model.state_dict(),
        'best_valid_score'  : max_dev_map,
        'optimizer'         : optimizer
    }
    torch.save(state, filename)

total_epochs = 4
best_dev_average_loss = None
for epoch in range(total_epochs):
    train_average_loss  = train_one()
    print('train_average_loss: {}'.format(train_average_loss))
    dev_average_loss    = eval_one()
    print('dev_average_loss: {}'.format(dev_average_loss))
    if (best_dev_average_loss is None or dev_average_loss >= best_dev_average_loss):
        best_dev_average_loss = dev_average_loss
        save_checkpoint(epoch, model, bert_model, best_dev_average_loss, optimizer, filename=os.path.join(odir, 'best_checkpoint.pth.tar'))



'''

# CUDA_VISIBLE_DEVICES=1 python3.6 train_bert_ir_embeds.py

'''

