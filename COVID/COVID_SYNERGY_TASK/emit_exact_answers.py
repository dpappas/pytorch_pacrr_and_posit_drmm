
from transformers import AutoTokenizer, AutoModel
import zipfile, json, pickle, random, os
from pprint import pprint
from tqdm import tqdm
import numpy as np
from pprint import pprint
from nltk.tokenize import sent_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

import logging, re
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as sklearn_auc

def pre_rec_auc(target, preds):
    # Data to plot precision - recall curve
    precision, recall, thresholds = precision_recall_curve(target, preds)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = sklearn_auc(recall, precision)
    # print(auc_precision_recall)
    return auc_precision_recall

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(level=logging.ERROR)

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def prep_bpe_data_tokens(tokens):
    sent_ids    = []
    for token in tokens:
        token_ids = bert_tokenizer.encode(token)[1:-1]
        sent_ids.extend(token_ids)
    ###################################################################
    sent_ids    = [bert_tokenizer.cls_token_id] + sent_ids + [bert_tokenizer.sep_token_id]
    return sent_ids

def prep_bpe_data_text(text):
    sent_ids = bert_tokenizer.encode(text)
    return sent_ids

def rebuild_tokens_from_bpes(bert_bpes):
  _tokens = []
  for bpe in bert_bpes:
    if bpe.startswith('##') :
      _tokens[-1] = _tokens[-1]+bpe[2:]
    else:
      _tokens.append(bpe)
  return _tokens

def pull_per_tokens(bert_bpe_ids, vecs, tags):
  ################################################################
  bert_bpes = bert_tokenizer.convert_ids_to_tokens(bert_bpe_ids)
  first_sep = bert_bpes.index('[SEP]')
  ################################################################
  _tokens  = []
  _vecs    = []
  _tags    = []
  ################################################################
  for i in range(first_sep+1, len(bert_bpes)-1):
    bpe = bert_bpes[i]
    vec = vecs[i]
    tag = tags[i]
    if bpe.startswith('##') :
      _tokens[-1] = _tokens[-1]+bpe[2:]
    else:
      _tokens.append(bpe)
      _vecs.append(vec)
      _tags.append(tag)
  return _tokens, _vecs, _tags

def load_model_from_checkpoint(resume_from):
    global start_epoch, optimizer
    if os.path.isfile(resume_from):
        # print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        my_model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))
    else:
        print("=> could not find path !!! '{}'".format(resume_from))

class Ontop_Modeler(nn.Module):
    def __init__(self, input_size, hidden_nodes):
        super(Ontop_Modeler, self).__init__()
        self.input_size             = input_size
        self.linear1                = nn.Linear(input_size, hidden_nodes, bias=True)
        self.linear2                = nn.Linear(hidden_nodes, 2, bias=True)
        self.loss                   = nn.BCELoss()
        self.tanh                   = nn.Tanh()
    def forward(self, input_xs):
        y = self.linear1(input_xs)
        y = self.tanh(y)
        y = self.linear2(y)
        return y

def fix_bert_tokens(tokens):
    ret = []
    for t in tokens:
        if (t.startswith('##')):
            ret[-1] = ret[-1] + t[2:]
        else:
            ret.append(t)
    return ret


my_seed = 1
random.seed(my_seed)
torch.manual_seed(my_seed)

use_cuda        = torch.cuda.is_available()
device          = torch.device("cuda") if(use_cuda) else torch.device("cpu")

model_name      = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"


bert_tokenizer 	= AutoTokenizer.from_pretrained(model_name, cache_dir='./')
pprint(bert_tokenizer.special_tokens_map)
bert_model 		= AutoModel.from_pretrained(model_name, cache_dir='./').to(device)
bert_model.eval()

epoch = 10
my_model_path = '/home/dpappas/bioasq_factoid/ktrapeznikov_biobert_v1.1_pubmed_squad_v2_MLP_{}.pth.tar'.format(epoch-1)
my_model        = Ontop_Modeler(768, 100).to(device)
load_model_from_checkpoint(my_model_path)
gb = my_model.eval()
#########################

def emit_exact_answers(qtext, snip):
    max_span        = 6
    ##########################################################
    sent_ids        = prep_bpe_data_text(snip.lower())[1:]
    sent_bpes       = bert_tokenizer.convert_ids_to_tokens(sent_ids)
    quest_ids       = prep_bpe_data_text(qtext.lower())
    ##########################################################
    bert_input      = torch.tensor([quest_ids+sent_ids]).to(device)
    ##########################################################
    bert_out        = bert_model(bert_input)[0]
    ##########################################################
    y               = my_model(bert_out)
    y               = torch.sigmoid(y).cpu()
    begin_y         = y[0, -len(sent_ids):, 0].data.numpy()
    end_y           = y[0, -len(sent_ids):, 1].data.numpy()
    ##########################################################
    ret = []
    for i in range(len(sent_bpes)):
        # print(sent_bpes[i], round(begin_y[i], 2), round(end_y[i], 2))
        if(begin_y[i] >= 0.5):
            start_ind       = i
            end_ind         = start_ind + np.argmax(end_y[i:i+max_span])
            answer_bpes     = sent_bpes[start_ind:end_ind+1]
            fixed_tokens    = fix_bert_tokens(answer_bpes)
            ret.append((' '.join(fixed_tokens), begin_y[i], end_y[end_ind]))
    if(len(ret) == 0):
        ##################################################### add best BEGIN
        start_ind       = np.argmax(begin_y)
        end_ind         = start_ind + np.argmax(end_y[start_ind:start_ind+max_span])
        answer_bpes     = sent_bpes[start_ind:end_ind+1]
        fixed_tokens    = fix_bert_tokens(answer_bpes)
        ret.append((' '.join(fixed_tokens), begin_y[start_ind], end_y[end_ind]))
        ##################################################### add best END
        end_ind         = np.argmax(end_y)
        start_ind       = end_ind - max_span + np.argmax(begin_y[max([0, end_ind-max_span]):end_ind+1])
        start_ind       = max([0, start_ind])
        answer_bpes     = sent_bpes[start_ind:end_ind+1]
        fixed_tokens    = fix_bert_tokens(answer_bpes)
        ret.append((' '.join(fixed_tokens), begin_y[start_ind], end_y[end_ind]))
    #########################################################
    return ret

if __name__ == '__main__':
    qtext   = 'what is the origin of COVID-19'
    snips   = [
        'In order to be approved to conduct saliva testing, based on regulatory guidelines at the time, we were required to compare paired NP and saliva collections from the same individuals, not only to validate saliva as an acceptable specimen type on our instrument.',
        'title'
    ]
    for snip in snips:
        res = emit_exact_answers(qtext, snip)
        pprint(res)






