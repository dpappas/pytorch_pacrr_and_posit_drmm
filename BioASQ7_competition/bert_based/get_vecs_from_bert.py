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

def embed_the_sents(sents, questions):
    ##########################################################################
    eval_examples       = []
    c = 0
    for sent, question in zip(sents, questions):
        eval_examples.append(InputExample(guid='example_dato_{}'.format(str(c)), text_a=sent, text_b=question, label=str(c)))
        c+=1
    ##########################################################################
    eval_features       = convert_examples_to_features(eval_examples, 256, bert_tokenizer)
    input_ids           = torch.tensor([ef.input_ids for ef in eval_features], dtype=torch.long).to(device)
    attention_mask      = torch.tensor([ef.input_mask for ef in eval_features], dtype=torch.long).to(device)
    ##########################################################################
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
    head_mask               = [None] * bert_model.config.num_hidden_layers
    token_type_ids          = torch.zeros_like(input_ids).to(device)
    embedding_output        = bert_model.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids)
    sequence_output, rest   = bert_model.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
    ##########################################################################
    if(adapt):
        first_token_tensors     = torch.stack([r[:, 0, :] for r in rest], dim=-1)
        print(first_token_tensors.size())
        weighted_vecs           = layers_weights(first_token_tensors).squeeze(-1)
        print(weighted_vecs.size())
    else:
        weighted_vecs           = sequence_output[:, 0, :]
    return weighted_vecs

use_cuda            = False
max_seq_length      = 50
device              = torch.device("cuda") if(use_cuda) else torch.device("cpu")
adapt               = True
layers_weights      = nn.Linear(13, 1, bias=False)
layers_weights.weight.data = torch.ones(13) / 13.
#####################

cache_dir           = 'bert-base-uncased' # '/home/dpappas/bert_cache/'
bert_tokenizer      = BertTokenizer.from_pretrained(cache_dir)
bert_model          = BertModel.from_pretrained(cache_dir,  output_hidden_states=True, output_attentions=False).to(device)
for param in bert_model.parameters():
    param.requires_grad = False

sents     = ['i am happy', 'i am unwell']
questions = ['are you happy ?', 'are you happy ?']

def fix_bert_tokens(tokens):
    ret = []
    for t in tokens:
        if (t.startswith('##')):
            ret[-1] = ret[-1] + t[2:]
        else:
            ret.append(t)
    return ret

def embed_the_sents_tokens(sents, questions=None):
    ##########################################################################
    if(questions is None):
        questions = [None] * len(sents)
    eval_examples       = []
    c = 0
    for sent, question in zip(sents, questions):
        eval_examples.append(InputExample(guid='example_dato_{}'.format(str(c)), text_a=sent, text_b=question, label=str(c)))
        c+=1
    ##########################################################################
    eval_features           = convert_examples_to_features(eval_examples, 256, bert_tokenizer)
    input_ids               = torch.tensor([ef.input_ids for ef in eval_features], dtype=torch.long).to(device)
    attention_mask          = torch.tensor([ef.input_mask for ef in eval_features], dtype=torch.long).to(device)
    ##########################################################################
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
    head_mask               = [None] * bert_model.config.num_hidden_layers
    token_type_ids          = torch.zeros_like(input_ids).to(device)
    embedding_output        = bert_model.embeddings(input_ids, position_ids=None, token_type_ids=token_type_ids)
    sequence_output, rest   = bert_model.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
    rest                    = torch.stack(rest, dim=-1)
    ##########################################################################
    if(adapt):
        rest = layers_weights(rest).squeeze(-1)
    else:
        rest = sequence_output
    ret_tokens, ret_vecs = [], []
    for i in range(len(sents)):
        bpes     = eval_features[i].tokens
        bpes     = bpes[:bpes.index('[SEP]')]
        tok_inds = [i for i in range(len(bpes)) if (not bpes[i].startswith('##') and bpes[i] not in ['[CLS]', '[SEP]'])]
        embeds   = rest[i][tok_inds]
        fixed_tokens = [ tok for tok in fix_bert_tokens(bpes) if tok not in ['[CLS]', '[SEP]']]
        ret_tokens.append(fixed_tokens)
        ret_vecs.append(embeds)
    return ret_tokens, ret_vecs

ret_tokens, ret_vecs = embed_the_sents_tokens(sents, questions)




optimizer_1     = optim.Adam(layers_weights.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for i in range(10):
    optimizer_1.zero_grad()
    v               = embed_the_sents(sents, questions)
    loss            = (v[0] * v[1]).sum()
    print(loss)
    loss.backward()
    optimizer_1.step()

# a z q
