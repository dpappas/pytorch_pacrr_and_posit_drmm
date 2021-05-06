
# We just use bart

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json, random, re
from pprint import pprint
from collections import Counter
from tqdm import tqdm
import torch
from nltk.tokenize import sent_tokenize
import sys

bioclean_mod    = lambda t: re.sub('[~`@#$-=<>/.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').replace("\n", ' ').strip().lower())

use_cuda        = torch.cuda.is_available()
device          = torch.device("cuda") if(use_cuda) else torch.device("cpu")

my_seed = 1
random.seed(my_seed)
torch.manual_seed(my_seed)

tokenizer     = AutoTokenizer.from_pretrained("Primer/bart-squad2")
model         = AutoModelForQuestionAnswering.from_pretrained("Primer/bart-squad2")
model.to('cuda')
model.eval()

b               = sys.argv[1]
fpath           = '/home/dpappas/bioasq_2021/BioASQ-task9bPhaseB-testset{}'.format(b)
ofpath          = '/home/dpappas/bioasq_2021/batch{}_system_4_factoid.json'.format(b)
d               = json.load(open(fpath))

def fix_phrase(phr):
    if len(phr) == 0:
        return ''
    while not phr[0].isalnum():
        phr = phr[1:]
    while not phr[-1].isalnum():
        phr = phr[:-1]
    return phr

def process_tuple(question, text):
    with torch.no_grad():
        seq             = '<s>' +  question + ' </s> ' + text + ' </s>'
        tokens          = tokenizer.encode_plus(seq, return_tensors='pt')
        input_ids       = tokens['input_ids'].to('cuda')
        attention_mask  = tokens['attention_mask'].to('cuda')
        index_of_sep      = input_ids[0].tolist().index(tokenizer.sep_token_id)
        ###################################################################
        out             = model(input_ids, attention_mask=attention_mask)
        start, end      = out[0], out[1]
        ###################################################################
        start, end      = start[:,index_of_sep+1:], end[:,index_of_sep+1:]
        input_ids       = input_ids[:,index_of_sep+1:]
        ###################################################################
        start_idx       = int(start.argmax().int())
        end_idx         = int(end.argmax().int())
        while(tokenizer.decode(input_ids[0, end_idx]).endswith('-')):
            end_idx += 1
        ###################################################################
        le_text         = tokenizer.decode(input_ids[0, start_idx:end_idx+1]).strip()
    return le_text

for quest in d['questions']:
    q_type = quest['type']
    if(q_type not in ['list', 'factoid']):
        continue
    q_body  = quest['body']
    q_id    = quest['id']
    print(q_body)
    answers = []
    for snip in quest['snippets']:
        for text in sent_tokenize(snip['text']):
            # text    = snip['text']
            ans     = process_tuple(q_body, text).lower()
            ans     = fix_phrase(ans)
            if len(ans) <3 or len(ans.split())>5:
                continue
            answers.append(ans)
    # answers = [ph for ph in answers if (check_answer(ph[0], quest['body']))]
    pprint(Counter(answers))
    print(40 * '=')
    answers = sorted(
        list(set(answers)),
        key=lambda x: answers.count(x),  # or max
        reverse=True
    )
    # ########################################
    if (quest['type'] == 'list'):
        quest['exact_answer'] = [[ans] for ans in answers]
    else :
        quest['exact_answer'] = [ans for ans in answers]

with open(ofpath, 'w') as of:
    of.write(json.dumps(d, indent=4, sort_keys=True))
    of.close()


'''
source ~/venvs/finetune_transformers/bin/activate
CUDA_VISIBLE_DEVICES=1 python factoid_system_4.py 5
'''

















