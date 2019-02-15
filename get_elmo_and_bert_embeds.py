from tqdm import tqdm
import pickle, json, re, os
from pprint import pprint
from nltk.tokenize import sent_tokenize
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from nltk.tokenize import sent_tokenize
from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
from allennlp.modules.elmo import Elmo, batch_to_ids
from joblib import Parallel, delayed
import random

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

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
    for parameter in model.parameters():
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

def read_examples(sentences):
    """Read a list of `InputExample`s from an input file."""
    ret = []
    for sent in sentences:
        line = sent.strip()
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
            text_b = None
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        unique_id = 0
        ret.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return ret

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
        # tokens_a = bioclean(example.text_a).split()
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # tokens_b = bioclean(example.text_b).split()
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
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
        # print(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        # if ex_index < 5:
        #     print("*** Example ***")
        #     print("unique_id: %s" % (example.unique_id))
        #     print("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     print("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids
            )
        )
    return features

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

def load_all_data(dataloc):
    print('loading pickle data')
    #
    with open(dataloc + 'BioASQ-trainingDataset6b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict((q['id'], q) for q in bioasq6_data['questions'])
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
    #
    train_data = RemoveBadYears(train_data, train_docs, True)
    train_data = RemoveTrainLargeYears(train_data, train_docs)
    dev_data = RemoveBadYears(dev_data, dev_docs, False)
    test_data = RemoveBadYears(test_data, test_docs, False)
    #
    return test_data, test_docs, dev_data, dev_docs, train_data, train_docs, bioasq6_data

def bert_embed_sents(sentences):
    if (len(sentences) == 0):
        return [], [], []
    # pprint(sentences)
    examples = read_examples(sentences)
    features = convert_examples_to_features(examples=examples, seq_length=100, tokenizer=tokenizer)
    all_ret_embeds = []
    bert_split_tokens = []
    bert_original_embeds = []
    for f in features:
        input_ids = torch.tensor([f.input_ids], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask], dtype=torch.long)
        tokens = f.tokens
        input_ids = input_ids[:, :input_mask.sum()]
        input_mask = input_mask[:, :input_mask.sum()]
        #
        all_encoder_layers, _ = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask,
                                      output_all_encoded_layers=True)
        my_embeddings = all_encoder_layers[-2].squeeze(0)
        #
        bert_split_tokens.append(tokens)
        bert_original_embeds.append(my_embeddings)
        #
        ret_embedds = []
        c = 0
        for t in list(zip(tokens, my_embeddings))[1:-1]:
            if (t[0].startswith('##')):
                ret_embedds[-1] += t[1]
                c += 1
            else:
                if (len(ret_embedds) > 0):
                    ret_embedds[-1] = ret_embedds[-1] / float(c)
                c = 1
                ret_embedds.append(t[1])
        ret_embedds = torch.stack(ret_embedds).cpu().detach().numpy()
        all_ret_embeds.append(ret_embedds)
    return all_ret_embeds, bert_split_tokens, bert_original_embeds

def tokenize(x):
    return bioclean(x)

def get_elmo_embeds(sentences):
    if (len(sentences) == 0):
        return []
    sentences       = [tokenize(s) for s in sentences if (len(s) > 0)]
    character_ids   = batch_to_ids(sentences)
    embeddings      = elmo(character_ids)
    the_embeds      = embeddings['elmo_representations'][0]
    ret             = [
        the_embeds[i, :len(sentences[i]), :].data.numpy()
        for i in range(len(sentences))
    ]
    return ret

def work(datum):
    question_bert_embeds = [' '.join(bioclean(datum['query_text']))]
    all_ret_embeds, bert_split_tokens, bert_original_embeds = bert_embed_sents(question_bert_embeds)
    datum['quest_bert_average_embeds'] = all_ret_embeds
    datum['quest_bert_original_embeds'] = bert_original_embeds
    datum['quest_bert_original_tokens'] = bert_split_tokens
    datum['quest_elmo_embeds'] = get_elmo_embeds(question_bert_embeds)
    return datum

def work2(args):
    doc, datum = args
    title_sents = sent_tokenize(datum['title'])
    title_sents = [' '.join(bioclean(s)) for s in title_sents]
    title_sents = [s for s in title_sents if (len(s.strip()) > 0)]
    #
    abs_sents = sent_tokenize(datum['abstractText'])
    abs_sents = [' '.join(bioclean(s)) for s in abs_sents]
    abs_sents = [s for s in abs_sents if (len(s.strip()) > 0)]
    #
    all_mesh = datum['meshHeadingsList']
    all_mesh = [' '.join(bioclean(m.replace(':', ' ', 1))) for m in all_mesh]
    all_mesh = ['mesh'] + all_mesh
    all_mesh = [s for s in all_mesh if (len(s.strip()) > 0)]
    #
    datum['title_sent_elmo_embeds'] = get_elmo_embeds(title_sents)
    datum['abs_sent_elmo_embeds'] = get_elmo_embeds(abs_sents)
    datum['mesh_elmo_embeds'] = get_elmo_embeds(all_mesh)
    #
    all_ret_embeds, bert_split_tokens, bert_original_embeds = bert_embed_sents(title_sents)
    datum['title_bert_average_embeds'] = all_ret_embeds
    datum['title_bert_original_embeds'] = bert_original_embeds
    datum['title_bert_original_tokens'] = bert_split_tokens
    #
    all_ret_embeds, bert_split_tokens, bert_original_embeds = bert_embed_sents(abs_sents)
    datum['abs_bert_average_embeds'] = all_ret_embeds
    datum['abs_bert_original_embeds'] = bert_original_embeds
    datum['abs_bert_original_tokens'] = bert_split_tokens
    #
    all_ret_embeds, bert_split_tokens, bert_original_embeds = bert_embed_sents(all_mesh)
    datum['mesh_bert_average_embeds'] = all_ret_embeds
    datum['mesh_bert_original_embeds'] = bert_original_embeds
    datum['mesh_bert_original_tokens'] = bert_split_tokens
    return (doc, datum)

def work3(args):
    doc, datum, odir = args
    #
    opath = os.path.join(odir, doc + '.p')
    if (not os.path.exists(opath)):
        title_sents = sent_tokenize(datum['title'])
        title_sents = [' '.join(bioclean(s.replace('\ufeff', ' '))) for s in title_sents]
        title_sents = [s for s in title_sents if (len(s.strip()) > 0)]
        #
        abs_sents = sent_tokenize(datum['abstractText'])
        abs_sents = [' '.join(bioclean(s.replace('\ufeff', ' '))) for s in abs_sents]
        abs_sents = [s for s in abs_sents if (len(s.strip()) > 0)]
        #
        all_mesh = datum['meshHeadingsList']
        all_mesh = [' '.join(bioclean(m.replace(':', ' ', 1).replace('\ufeff', ' '))) for m in all_mesh]
        all_mesh = ['mesh'] + all_mesh
        all_mesh = [s for s in all_mesh if (len(s.strip()) > 0)]
        #
        ret = {}
        #
        ret['title_sent_elmo_embeds'] = get_elmo_embeds(title_sents)
        ret['abs_sent_elmo_embeds'] = get_elmo_embeds(abs_sents)
        ret['mesh_elmo_embeds'] = get_elmo_embeds(all_mesh)
        #
        all_ret_embeds, bert_split_tokens, bert_original_embeds = bert_embed_sents(title_sents)
        ret['title_bert_average_embeds'] = all_ret_embeds
        ret['title_bert_original_embeds'] = bert_original_embeds
        ret['title_bert_original_tokens'] = bert_split_tokens
        #
        all_ret_embeds, bert_split_tokens, bert_original_embeds = bert_embed_sents(abs_sents)
        ret['abs_bert_average_embeds'] = all_ret_embeds
        ret['abs_bert_original_embeds'] = bert_original_embeds
        ret['abs_bert_original_tokens'] = bert_split_tokens
        #
        all_ret_embeds, bert_split_tokens, bert_original_embeds = bert_embed_sents(all_mesh)
        ret['mesh_bert_average_embeds'] = all_ret_embeds
        ret['mesh_bert_original_embeds'] = bert_original_embeds
        ret['mesh_bert_original_tokens'] = bert_split_tokens
        #
        pickle.dump(ret, open(opath, 'wb'))

# # laptop
# init_checkpoint_pt  = "/home/dpappas/for_ryan/uncased_L-12_H-768_A-12/"
# dataloc             = '/home/dpappas/for_ryan/'
# options_file        = "/home/dpappas/for_ryan/elmo_weights/options.json"
# weight_file         = "/home/dpappas/for_ryan/elmo_weights/weights.hdf5"

# atlas , cslab243
init_checkpoint_pt  = "/home/dpappas/bioasq_all/uncased_L-12_H-768_A-12/"
dataloc             = '/home/dpappas/bioasq_all/bioasq_data/'
options_file        = "/home/dpappas/bioasq_all/elmo_weights/options.json"
weight_file         = "/home/dpappas/bioasq_all/elmo_weights/weights.hdf5"
odir                = "/home/dpappas/bioasq_all/bert_elmo_embeds/"

if (not os.path.exists(odir)):
    os.makedirs(odir)

cache_dir   = '/home/dpappas/bert_cache/'
if(not os.path.exists(cache_dir)):
    os.makedirs(cache_dir)

tokenizer   = BertTokenizer.from_pretrained(
    # pretrained_model_name='bert-large-uncased',
    pretrained_model_name='/home/dpappas/bert_cache/bert-large-uncased-vocab.txt',
    cache_dir=cache_dir
)

(test_data, test_docs, dev_data, dev_docs, train_data, train_docs, bioasq6_data) = load_all_data(dataloc=dataloc)

if (not os.path.exists(os.path.join(init_checkpoint_pt, 'pytorch_model.bin'))):
    convert_tf_checkpoint_to_pytorch(
        os.path.join(init_checkpoint_pt, 'bert_model.ckpt'),
        os.path.join(init_checkpoint_pt, 'bert_config.json'),
        os.path.join(init_checkpoint_pt, 'pytorch_model.bin')
    )

elmo    = Elmo(options_file, weight_file, 1, dropout=0)
model   = BertModel.from_pretrained(init_checkpoint_pt, cache_dir=cache_dir)
# model               = model.cuda()
model.eval()

nof_threads = 16

#######################################################

# test_data['queries']    = Parallel(n_jobs=nof_threads, verbose=0, backend="threading")(map(delayed(work), tqdm(test_data['queries'])))
# with open(dataloc + 'bioasq_bm25_top100_bert_elmo.test.pkl', 'wb') as f:
#     pickle.dump(test_data, f)
#     f.close()
#
# dev_data['queries']     = Parallel(n_jobs=nof_threads, verbose=0, backend="threading")(map(delayed(work), tqdm(dev_data['queries'])))
# with open(dataloc + 'bioasq_bm25_top100_bert_elmo.dev.pkl', 'wb') as f:
#     pickle.dump(dev_data , f)
#     f.close()
#
# train_data['queries']   = Parallel(n_jobs=nof_threads, verbose=0, backend="threading")(map(delayed(work), tqdm(train_data['queries'])))
# with open(dataloc + 'bioasq_bm25_top100_bert_elmo.train.pkl', 'wb') as f:
#     pickle.dump(train_data , f)
#     f.close()

#######################################################

# dev_docs                = dict(Parallel(n_jobs=nof_threads, verbose=0, backend="threading")(map(delayed(work2), tqdm(list(dev_docs.items())))))
# with open(dataloc + 'bioasq_bm25_docset_top100_bert_elmo.dev.pkl', 'wb') as f:
#     pickle.dump(dev_docs, f)
#     f.close()
#
# test_docs               = dict(Parallel(n_jobs=nof_threads, verbose=0, backend="threading")(map(delayed(work2), tqdm(list(test_docs.items())))))
# with open(dataloc + 'bioasq_bm25_docset_top100_bert_elmo.test.pkl', 'wb') as f:
#     pickle.dump(test_docs, f)
#     f.close()
#
# train_docs              = dict(Parallel(n_jobs=nof_threads, verbose=0, backend="threading")(map(delayed(work2), tqdm(list(train_docs.items())))))
# with open(dataloc + 'bioasq_bm25_docset_top100_bert_elmo.train.pkl', 'wb') as f:
#     pickle.dump(train_docs, f)
#     f.close()

#######################################################

the_docs    = dev_docs
total       = len(the_docs.keys())
for doc in tqdm(random.sample(the_docs.keys(), len(the_docs.keys()))):
    work3((doc, the_docs[doc], odir))

del (dev_docs)
print('\nDone dev\n')

the_docs = test_docs
total += len(the_docs.keys())
for doc in tqdm(random.sample(the_docs.keys(), len(the_docs.keys()))):
    work3((doc, the_docs[doc], odir))

del (test_docs)
print('\nDone test\n')

the_docs = train_docs
total += len(the_docs.keys())
for doc in tqdm(random.sample(the_docs.keys(), len(the_docs.keys()))):
    work3((doc, the_docs[doc], odir))

del (train_docs)
print('\nDone train\n')
print('total: {}'.format(total))

if(not os.path.exists('/home/dpappas/bioasq_all/all_quest_embeds.p')):
    all_qs = {}
    for t in tqdm(test_data['queries'] + train_data['queries'] + dev_data['queries'], ascii=True):
        text = t['query_text']
        quest_sents = sent_tokenize(text)
        quest_sents = [' '.join(bioclean(s.replace('\ufeff', ' '))) for s in quest_sents]
        quest_sents = [s for s in quest_sents if (len(s.strip()) > 0)]
        #
        all_qs[text] = {}
        all_qs[text]['title_sent_elmo_embeds'] = get_elmo_embeds(quest_sents)
        all_ret_embeds, bert_split_tokens, bert_original_embeds = bert_embed_sents(quest_sents)
        all_qs[text]['title_bert_average_embeds'] = all_ret_embeds
        all_qs[text]['title_bert_original_embeds'] = bert_original_embeds
        all_qs[text]['title_bert_original_tokens'] = bert_split_tokens
    pickle.dump(all_qs, open('/home/dpappas/bioasq_all/all_quest_embeds.p', 'wb'))

print('Good to go!!!')

'''
manolis mail gia text analytics     -   OK
read the thread for stat. sign.     -   pending

snippets: 
MAP + NDCG                          -   pending
stat. sign.                         -   pending

doc ret.
BM25 baseline                       -   pending
stat. sign.                         -   pending


joint model
voting 5 modelwn (ensemble)         -   pending
stat. sign.                         -   pending


Bert+Elmo -> PCA                    -   pending
Baselines se bert ???               -   pending

BM25+ABCNN3                         -   pending
BM25+PDRMM                          -   pending

use setting 1 and not setting 3 in pipeline and joint models

'''

'''
Apo to bert pairnw to proteleftaio layer
Apo to elmo apla kalw to modelo apo to standarad implementation pou mou dinei to teleftaio layer
'''
