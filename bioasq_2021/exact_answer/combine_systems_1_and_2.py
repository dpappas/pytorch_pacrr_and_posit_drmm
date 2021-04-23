
'''

dpappas@atlas ~/bioasq_factoid python3.6 eval_mlp_small.py \
"ktrapeznikov/biobert_v1.1_pubmed_squad_v2" \
"ktrapeznikov_biobert_v1.1_pubmed_squad_v2_MLP_100_8.pth.tar"

system 1 : snipBefAfter1_ktrapeznikov__biobert_v1.1_pubmed_squad_v2_MLP_100_9.pth.tar
system 2 : ktrapeznikov_biobert_v1.1_pubmed_squad_v2_MLP_100_8.pth.tar

'''


from collections import Counter
from transformers import AutoTokenizer, AutoModel
import zipfile, json, pickle, random, os, sys, re
from pprint import pprint
from tqdm import tqdm
import numpy as np
from pprint import pprint
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize

def first_alpha_is_upper(sent):
    # specials = [
    #     '__EU__','__SU__','__EMS__','__SMS__','__SI__',
    #     '__ESB','__SSB__','__EB__','__SB__','__EI__',
    #     '__EA__','__SA__','__SQ__','__EQ__','__EXTLINK',
    #     '__XREF','__URI', '__EMAIL','__ARRAY','__TABLE',
    #     '__FIG','__AWID','__FUNDS', '__sup__', '__end_sup__',
    #     '__sub__', '__end_sub__', '__i_tag__', '__end_i_tag__',
    #     '__underline__', '__end_underline__', '__bold__', '__end_bold__',
    #     '__italic__', '__end_italic__' , ' __EU__ ' , ' __SU__ ' , ' __ESP__ ' , ' __SSP__ ',
    #     ' __ESB__ ',' __SSB__ ',' __EB__ ',' __SB__ ', ' __EI__ ', ' __SI__ ','__SQ__', '__EQ__', '__EXTLINK{}__')
    # tr.replace_keep_case('<ext-link.*?/>', '__EXTLINK{}__')
    # tr.replace_keep_case('<xref.*?>.*?</xref>', '__XREF{}__')
    # tr.replace_keep_case('<uri.*?</uri>', '__URI{}__')
    # tr.replace_keep_case('<media.*?</media>', '__MEDIA{}__')
    #
    # ]
    # for special in specials:
    #     sent = sent.replace(special,'')
    sent = ' '.join(
        [
            tok
            for tok in sent.split()
            if not tok.startswith('__')
            and tok.isalnum()
        ]
    )
    for c in sent:
        if(c.isalpha()):
            if(c.isupper()):
                return True
            else:
                return False
    return False

def ends_with_special(sent):
    sent = sent.lower()
    ind = [item.end() for item in re.finditer('[\W\s]sp.|[\W\s]nos.|[\W\s]figs.|[\W\s]sp.[\W\s]no.|[\W\s][vols.|[\W\s]cv.|[\W\s]fig.|[\W\s]e.g.|[\W\s]et[\W\s]al.|[\W\s]i.e.|[\W\s]p.p.m.|[\W\s]cf.|[\W\s]n.a.|[\W\s]min.', sent)]
    if(len(ind)==0):
        return False
    else:
        ind = max(ind)
        if (len(sent) == ind):
            return True
        else:
            return False

def starts_with_special(sent):
    sent    = sent.strip().lower()
    chars   = ':%@#$^&*()\\,<>?/=+-_'
    for c in chars:
        if(sent.startswith(c)):
            return True
    return False

def split_sentences2(text):
    sents = [l.strip() for l in sent_tokenize(text)]
    ret = []
    i = 0
    while (i < len(sents)):
        sent = sents[i]
        while (
            ((i + 1) < len(sents)) and
            (
                ends_with_special(sent) or
                not first_alpha_is_upper(sents[i+1]) or
                starts_with_special(sents[i + 1])
                # sent[-5:].count('.') > 1       or
                # sents[i+1][:10].count('.')>1   or
                # len(sent.split()) < 2          or
                # len(sents[i+1].split()) < 2
            )
        ):
            sent += ' ' + sents[i + 1]
            i += 1
        ret.append(sent.replace('\n', ' ').strip())
        i += 1
    return ret

def get_sents(ntext):
    sents = []
    for subtext in ntext.split('\n'):
        subtext = re.sub('\s+', ' ', subtext.replace('\n',' ')).strip()
        if (len(subtext) > 0):
            ss = split_sentences2(subtext)
            sents.extend([ s for s in ss if(len(s.strip())>0)])
    if(len(sents)>0 and len(sents[-1]) == 0 ):
        sents = sents[:-1]
    return sents

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

def load_model_from_checkpoint(resume_from, the_model):
    global start_epoch, optimizer
    if os.path.isfile(resume_from):
        # print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        the_model.load_state_dict(checkpoint['state_dict'])
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

my_seed = 1
random.seed(my_seed)
torch.manual_seed(my_seed)

use_cuda        = torch.cuda.is_available()
device          = torch.device("cuda") if(use_cuda) else torch.device("cpu")

b               = sys.argv[1]
fpath           = '/home/dpappas/bioasq_2021/BioASQ-task9bPhaseB-testset{}'.format(b)
ofpath          = '/home/dpappas/bioasq_2021/batch{}_system_3_factoid.json'.format(b)

model_name      = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"
my_model_path_1 = '/home/dpappas/bioasq_factoid/snipBefAfter1_ktrapeznikov__biobert_v1.1_pubmed_squad_v2_MLP_100_9.pth.tar'
my_model_path_2 = '/home/dpappas/bioasq_factoid/ktrapeznikov_biobert_v1.1_pubmed_squad_v2_MLP_100_8.pth.tar'
d               = json.load(open(fpath))

bert_tokenizer 	= AutoTokenizer.from_pretrained(model_name)
pprint(bert_tokenizer.special_tokens_map)
bert_model 		= AutoModel.from_pretrained(model_name).to(device)
bert_model.eval()

my_model_1      = Ontop_Modeler(768, 100).to(device)
load_model_from_checkpoint(my_model_path_1, my_model_1)
gb              = my_model_1.eval()

my_model_2      = Ontop_Modeler(768, 100).to(device)
load_model_from_checkpoint(my_model_path_2, my_model_2)
gb              = my_model_2.eval()

def fix_phrase(phr):
    if len(phr) == 0:
        return ''
    while not phr[0].isalnum():
        phr = phr[1:]
    while not phr[-1].isalnum():
        phr = phr[:-1]
    return phr

def approach_1(begin_y, end_y, sent_ids, prob_thresh=0.1):
    answers =[]
    if(begin_y.max() < prob_thresh and end_y.max() < prob_thresh):
        return []
    ##########################################################
    plus_minus_tokens   = 14
    if(begin_y.max() >= end_y.max()):
        start_from      = begin_y.argmax().int()
        go_up_to        = start_from + plus_minus_tokens+1
        begin_y         = begin_y[start_from:go_up_to]
        end_y           = end_y[start_from:go_up_to]
        sent_ids        = sent_ids[start_from:go_up_to]
    else:
        go_up_to        = end_y.argmax().int() + 1
        start_from      = max([0, go_up_to - plus_minus_tokens - 2])
        begin_y         = begin_y[start_from:go_up_to]
        end_y           = end_y[start_from:go_up_to]
        sent_ids        = sent_ids[start_from:go_up_to]
    ##########################################################
    start_idx           = int(begin_y.argmax().int())
    end_idx             = int(end_y.argmax().int())
    ##########################################################
    # plus_minus_tokens   = 5
    # if(start_idx > end_idx): # ean to start einai meta to end
    #     if(begin_y[start_idx] > end_y[end_idx]):  # ean to skor tou start einai megalytero krataw  apo start mexri start+plus_minus_tokens
    #         to_         = start_idx+plus_minus_tokens+1
    #         end_idx     = start_idx + end_y[start_idx:to_].argmax()
    #     else:   # ean to skor tou end einai megalytero krataw apo end mexri end-plus_minus_tokens
    #         from_       = max([0, end_idx-plus_minus_tokens])
    #         from_       = plus_minus_tokens - begin_y[from_:end_idx+1].argmax()
    #         start_idx   = end_idx - max([0,from_])
    #         start_idx   = max([0, start_idx])
    # elif end_idx > start_idx + plus_minus_tokens:
    #     pass # not yet impl.
    ##########################################################
    ans                 = bert_tokenizer.decode(sent_ids[start_idx:end_idx+1])
    score               = (begin_y.max() + end_y.max()) / 2.0
    #########################################################
    # print(qtext)
    # print(text)
    # print(ans)
    # print(40 * '-')
    answers.append((ans, score.cpu().item()))
    return answers

def approach_2(begin_y, end_y, sent_ids, prob_thresh=0.1):
    answers =[]
    ##########################################################
    # begin_y[begin_y<0.5]    = 0
    # begin_y[begin_y>0.5]    = 1
    # end_y[end_y<0.5]        = 0
    # end_y[end_y>0.5]        = 1
    ##########################################################
    for i in range(len(begin_y)):
        if(begin_y[i] > prob_thresh):
            for j in range(i, len(end_y)):
                if(end_y[j] > prob_thresh):
                    ans     = bert_tokenizer.decode(sent_ids[i:j+1])
                    score   = (begin_y[i] + end_y[j]) / 2.0
                    answers.append((ans, score.cpu().item()))
                    break
    return answers

with torch.no_grad():
    for quest in d['questions']:
        q_type = quest['type']
        if(q_type not in ['list', 'factoid']):
            continue
        qtext   = quest['body']
        q_id    = quest['id']
        answers = []
        answers_accept_all = []
        for snip in quest['snippets']:
            for text in get_sents(snip['text']):
                if(len(text.split())<6):
                    continue
                # text = snip['text']
                # qtext = 'List the deadliest viruses in the world.'
                # text = 'WHO ranks HIV as one of the deadliest diseases'
                sent_ids        = prep_bpe_data_text(text.lower())[1:]
                quest_ids       = prep_bpe_data_text(qtext.lower())
                ##########################################################
                bert_input      = torch.tensor([quest_ids+sent_ids]).to(device)
                bert_out        = bert_model(bert_input)[0]
                ##########################################################
                y               = torch.sigmoid(my_model_1(bert_out))
                begin_y         = y[0, -len(sent_ids):, 0]
                end_y           = y[0, -len(sent_ids):, 1]
                answers.extend(approach_1(begin_y, end_y, sent_ids))
                answers_accept_all.extend(approach_1(begin_y, end_y, sent_ids, prob_thresh = 0.01))
                ##########################################################
                y               = torch.sigmoid(my_model_2(bert_out))
                begin_y         = y[0, -len(sent_ids):, 0]
                end_y           = y[0, -len(sent_ids):, 1]
                answers.extend(approach_1(begin_y, end_y, sent_ids))
                answers_accept_all.extend(approach_1(begin_y, end_y, sent_ids, prob_thresh = 0.01))
        #########################################################
        if(len(answers) == 0):
            answers.extend(answers_accept_all)
        print(qtext)
        pprint(answers)
        # pprint(Counter([t[0] for t in answers]))
        answers = sorted(
            list(set([t[0] for t in answers])),
            key=lambda x: sum([t[1] for t in answers if t[0] == x]), # or max
            reverse=True
        )
        pprint(answers)
        #########################################################
        if (quest['type'] == 'list'):
            quest['exact_answer'] = [[ans] for ans in answers]
        else :
            quest['exact_answer'] = [ans for ans in answers]
        #########################################################

with open(ofpath, 'w') as of:
    of.write(json.dumps(d, indent=4, sort_keys=True))
    of.close()

'''

source ~/venvs/finetune_transformers/bin/activate
CUDA_VISIBLE_DEVICES=1 python combine_systems_1_and_2.py 4

'''

'''
exit()

qtext   = "List as many European influenza vaccines as possible."
text    = "Three split-virion vaccines (Vaxigrip, Begrivac, and Influsplit/Fluarix) and three subunit vaccines containing only viral surface glycoproteins (Influvac, Agrippal, and Fluvirin) available for the 1994-95 season were analysed by biological, molecular, and biochemical methods."
sent_ids = prep_bpe_data_text(text.lower())[1:]
quest_ids = prep_bpe_data_text(qtext.lower())
bert_input = torch.tensor([quest_ids + sent_ids]).to(device)
bert_out = bert_model(bert_input)[0]
y = torch.sigmoid(my_model(bert_out))
begin_y         = y[0, -len(sent_ids):, 0]
end_y           = y[0, -len(sent_ids):, 1]
for i in range(len(sent_ids)):
    print(
        '\t'.join(
            [
                str(t) for t in
                [
                    '{:18s}'.format(bert_tokenizer.convert_ids_to_tokens(sent_ids[i])),
                    sent_ids[i],
                    round(begin_y[i].cpu().item(), 2),
                    round(end_y[i].cpu().item(), 2)
                ]
            ]
        )
    )

approach_1(begin_y, end_y, sent_ids, prob_thresh = 0.01)
approach_2(begin_y, end_y, sent_ids, prob_thresh = 0.01)
'''
