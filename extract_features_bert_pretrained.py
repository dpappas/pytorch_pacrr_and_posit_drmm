# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, random, pickle
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from pprint import pprint
import codecs
import collections
import json
import re
from bert import modeling, tokenization
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

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

def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn

def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu, use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    all_layers = model.get_all_encoder_layers()
    print('total_layers: {}'.format(len(all_layers)))
    predictions = {
        "unique_id": unique_ids,
    }

    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

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
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
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

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

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

    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features

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

def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1
  return examples

def get_bert_for_text(some_sents):
    examples = []
    unique_id = 0
    unique_id_to_text = {}
    for sent in some_sents:
        line                         = tokenization.convert_to_unicode(sent.strip()).strip()
        example                      = InputExample(unique_id=unique_id, text_a=line, text_b=None)
        unique_id_to_text[unique_id] = sent
        unique_id                    += 1
        examples.append(example)
    ####
    features    = convert_examples_to_features(examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)
    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature
    ####
    input_fn    = input_fn_builder(features=features, seq_length=max_seq_length)
    ####
    ret = []
    for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id   = int(result["unique_id"])
        feature     = unique_id_to_feature[unique_id]
        tokens      = feature.tokens
        aver_embeds = sum([result[k] for k in result.keys() if ('layer_' in k)])
        aver_embeds = aver_embeds[:len(tokens)]
        inds        = [i for i in range(len(tokens)) if(not tokens[i].startswith('##'))]
        sent_text   = unique_id_to_text[unique_id]
        ret.append((sent_text, tokens, inds, aver_embeds))
    ####
    return ret

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

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
####
dataloc             = '/home/dpappas/bioasq_all/bioasq_data/'
odir                = '/media/dpappas/dpappas_data/biobert_data/'
bert_config_file    = '/home/dpappas/bioasq_all/F_BERT/Biobert/pubmed_pmc_470k/bert_config.json'
init_checkpoint     = '/home/dpappas/bioasq_all/F_BERT/Biobert/pubmed_pmc_470k/biobert_model.ckpt'
vocab_file          = '/home/dpappas/bioasq_all/F_BERT/Biobert/pubmed_pmc_470k/vocab.txt'
####
if(not os.path.exists(odir)):
    os.makedirs(odir)

do_lower_case       = True
max_seq_length      = 100
layer_indexes       = [i for i in range(12)]
num_shards          = 8
predict_batch_size  = 8
#
bert_config         = modeling.BertConfig.from_json_file(bert_config_file)
tokenizer           = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
model_fn            = model_fn_builder(bert_config=bert_config, init_checkpoint=init_checkpoint, layer_indexes=layer_indexes, use_tpu=False, use_one_hot_embeddings=False)
is_per_host         = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config          = tf.contrib.tpu.RunConfig(master=None, tpu_config=tf.contrib.tpu.TPUConfig(num_shards=num_shards, per_host_input_for_training=is_per_host))
estimator           = tf.contrib.tpu.TPUEstimator(use_tpu=False, model_fn=model_fn, config=run_config, predict_batch_size=predict_batch_size)
#
# text                = '''A sulfated glycoprotein was isolated from the culture media of Drosophila Kc cells and named papilin. Affinity purified antibodies against this protein localized it primarily to the basement membranes of embryos. The antibodies cross-reacted with another material which was not sulfated and appeared to be the core protein of papilin, which is proteoglycan-like. After reduction, papilin electrophoresed in sodium dodecyl sulfate-polyacrylamide gel electrophoresis as a broad band of about 900,000 apparent molecular weight and the core protein as a narrow band of approximately 400,000. The core protein was formed by some cell lines and by other cells on incubation with 1 mM 4-methylumbelliferyl xyloside, which inhibited formation of the proteoglycan-like form. The buoyant density of papilin in CsCl/4 M guanidine hydrochloride is 1.4 g/ml, that of the core protein is much less. Papilin forms oligomers linked by disulfide bridges, as shown by sodium dodecyl sulfate-agarose gel electrophoresis and electron microscopy. The protomer is a 225 +/- 15-nm thread which is disulfide-linked into a loop with fine, protruding thread ends. Oligomers form clover-leaf-like structures. The protein contains 22% combined serine and threonine residues and 25% combined aspartic and glutamic residues. 10 g of polypeptide has attached 6.4 g of glucosamine, 3.1 g of galactosamine, 6.1 g of uronic acid, and 2.7 g of neutral sugars. There are about 80 O-linked carbohydrate chains/core protein molecule. Sulfate is attached to these chains. The O-linkage is through an unidentified neutral sugar. Papilin is largely resistant to common glycosidases and several proteases. The degree of sulfation varies with the sulfate concentration of the incubation medium. This proteoglycan-like glycoprotein differs substantially from corresponding proteoglycans found in vertebrate basement membranes, in contrast to Drosophila basement membrane laminin and collagen IV which have been conserved evolutionarily.'''.strip()
# bert_data           = get_bert_for_text(text)

(test_data, test_docs, dev_data, dev_docs, train_data, train_docs, bioasq6_data) = load_all_data(dataloc=dataloc)

for the_docs in [test_docs, dev_docs, train_docs]:
    total       = len(the_docs.keys())
    for doc in tqdm(random.sample(the_docs.keys(), len(the_docs.keys()))):
        opath = os.path.join(odir, doc + '.p')
        if (not os.path.exists(opath)):
            datum       = the_docs[doc]
            #
            title_sents = sent_tokenize(datum['title'])
            title_sents = [' '.join(bioclean(s.replace('\ufeff', ' '))) for s in title_sents]
            title_sents = [s for s in title_sents if (len(s.strip()) > 0)]
            #
            abs_sents = sent_tokenize(datum['abstractText'])
            abs_sents = [' '.join(bioclean(s.replace('\ufeff', ' '))) for s in abs_sents]
            abs_sents = [s for s in abs_sents if (len(s.strip()) > 0)]
            #
            ret = {}
            ret['abs_sents']                    = abs_sents
            ret['title_sents']                  = title_sents
            ret['title_bert_original_embeds']   = get_bert_for_text(title_sents)
            ret['abs_bert_original_embeds']     = get_bert_for_text(abs_sents)
            #
            pickle.dump(ret, open(opath, 'wb'), protocol=2)

oqf = '/media/dpappas/dpappas_data/biobert_data/all_quest_embeds.p'
if(not os.path.exists(oqf)):
    all_qs = {}
    for t in tqdm(test_data['queries'] + train_data['queries'] + dev_data['queries'], ascii=True):
        text        = t['query_text']
        quest_sents = sent_tokenize(text)
        quest_sents = [' '.join(bioclean(s.replace('\ufeff', ' '))) for s in quest_sents]
        quest_sents = [s for s in quest_sents if (len(s.strip()) > 0)]
        #
        all_qs[text] = {}
        all_qs[text]['title_bert_original_embeds'] = get_bert_for_text(quest_sents)
    pickle.dump(all_qs, open(oqf, 'wb'))

print('Good to go!!!')


'''
python3.6 \
/home/dpappas/PycharmProjects/pytorch_pacrr_and_posit_drmm/extract_features_bert_pretrained.py \
--input_file=./input.txt \
--output_file=./output.jsonl \
--vocab_file=/home/dpappas/Downloads/F_BERT/Biobert/pubmed_pmc_470k/vocab.txt \
--bert_config_file=/home/dpappas/Downloads/F_BERT/Biobert/pubmed_pmc_470k/bert_config.json \
--init_checkpoint=/home/dpappas/Downloads/F_BERT/Biobert/pubmed_pmc_470k/biobert_model.ckpt \
--layers=-1,-2,-3,-4 \
--max_seq_length=300 \
--batch_size=8

'''

