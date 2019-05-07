#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

import  os
import  json
import  time
import  random
import  logging
import  subprocess
import  torch
import  torch.nn.functional         as F
import  torch.nn                    as nn
import  numpy                       as np
import  torch.optim                 as optim
import  torch.autograd              as autograd
import  pickle
from    tqdm import tqdm
from    pprint import pprint
from    nltk.tokenize import sent_tokenize
from    difflib import SequenceMatcher
import  nltk
import  math
from    sklearn.preprocessing import LabelEncoder
from    sklearn.preprocessing import OneHotEncoder
import  re
import  logging
import  torch
from    pytorch_pretrained_bert.tokenization import BertTokenizer
from    torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from    torch.utils.data.distributed import DistributedSampler
from    pytorch_pretrained_bert.tokenization import BertTokenizer
from    pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertModel, BertForQuestionAnswering
from    pytorch_pretrained_bert.optimization import BertAdam
from    pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

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

def fix_bert_tokens(tokens):
    ret = []
    for t in tokens:
        if (t.startswith('##')):
            ret[-1] = ret[-1] + t[2:]
        else:
            ret.append(t)
    return ret

eval_examples   = [
    InputExample(
        guid='example_dato_1',
        text_a='i asuyubdwanna beasdasda the very best',
        text_b=None,
        label='1'
    )
]

use_cuda            = True
max_seq_length      = 50
device              = torch.device("cuda") if(use_cuda) else torch.device("cpu")
bert_model          = 'bert-base-uncased'
cache_dir           = '/home/dpappas/bert_cache/'

bert_tokenizer      = BertTokenizer.from_pretrained(bert_model, do_lower_case=True, cache_dir=cache_dir)
bert_model          = BertForQuestionAnswering.from_pretrained(bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1)).to(device)

eval_features       = convert_examples_to_features(eval_examples, max_seq_length, bert_tokenizer)
eval_feat           = eval_features[0]
input_ids           = torch.tensor([eval_feat.input_ids], dtype=torch.long).to(device)
input_mask          = torch.tensor([eval_feat.input_mask], dtype=torch.long).to(device)
segment_ids         = torch.tensor([eval_feat.segment_ids], dtype=torch.long).to(device)
tokens              = eval_feat.tokens
token_embeds, pooled_output = bert_model.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)

fixed_tokens        = fix_bert_tokens(eval_feat.tokens)

tok_inds            = [i for i in range(len(tokens)) if (not tokens[i].startswith('##'))]
token_embeds        = token_embeds.squeeze(0)
embs                = token_embeds[tok_inds, :]


