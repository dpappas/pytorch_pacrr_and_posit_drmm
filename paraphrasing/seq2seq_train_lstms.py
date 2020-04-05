
import torch, re, random, spacy, time, pickle
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch import optim
from torch import FloatTensor as FT
from my_data_handling import DataHandler
from pprint import pprint

my_seed = 1989
random.seed(my_seed)
torch.manual_seed(my_seed)

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

######################################################################################################
use_cuda    = torch.cuda.is_available()
use_cuda    = False
device      = torch.device("cuda") if(use_cuda) else torch.device("cpu")
######################################################################################################
en = spacy.load('en_core_web_sm')

def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

class SGNS(nn.Module):
    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding  = embedding
        self.vocab_size = vocab_size
        self.n_negs     = n_negs
        self.weights    = None
        if weights is not None:
            wf              = np.power(weights, 0.75)
            wf              = wf / wf.sum()
            self.weights    = FT(wf)
    def forward(self, true_vecs, out_vecs):
        batch_size      = true_vecs.size()[0]
        context_size    = true_vecs.size()[1]
        if self.weights is not None:
            nwords  = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords  = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long().to(device)
        nvectors    = self.embedding(nwords).neg()
        # print(out_vecs.size())
        # print(true_vecs.size())
        # print(nvectors.size())
        oloss       = torch.bmm(out_vecs, true_vecs.transpose(1,2))
        oloss       = oloss.sigmoid().log()
        oloss       = oloss.mean(1)
        nloss       = torch.bmm(nvectors, true_vecs.transpose(1,2))
        nloss       = nloss.squeeze().sigmoid().log()
        nloss       = nloss.view(-1, context_size, self.n_negs)
        nloss       = nloss.sum(2).mean(1)
        # print(oloss.size())
        # print(nloss.size())
        return -(oloss + nloss).mean()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_one(clip):
    model.train()
    epoch_loss, i   = 0, 0
    train_iterator  = data_handler.iter_train_batches(b_size)
    pbar            = tqdm(enumerate(train_iterator), total= int(data_handler.number_of_train_instances / b_size) +1)
    for i, batch in pbar:
        src         = torch.LongTensor(batch['src_ids'])
        trg         = torch.LongTensor(batch['trg_ids'])
        ##########################################
        optimizer.zero_grad()
        loss = model(src, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        ##########################################
        pbar.set_description('train_aver_loss batch {}: {}'.format(i, epoch_loss / float(i+1)))
    return epoch_loss / float(i+1)

def eval_one():
    model.eval()
    with torch.no_grad():
        epoch_loss, i   = 0, 0
        dev_iterator    = data_handler.iter_dev_batches(b_size)
        pbar            = tqdm(enumerate(dev_iterator), total= int(data_handler.number_of_dev_instances / b_size)+1)
        for i, batch in pbar:
            src         = torch.LongTensor(batch['src_ids'])
            trg         = torch.LongTensor(batch['trg_ids'])
            loss        = model(src, trg[:, :-1])
            epoch_loss += loss.item()
            pbar.set_description('eval_aver_loss batch {}: {}'.format(i, epoch_loss / float(i+1)))
    return epoch_loss / float(i+1)

class S2S_lstm(nn.Module):
    def __init__(self, vocab_size = 100, embedding_dim=30, hidden_dim = 256, src_pad_token=1, trg_pad_token=1):
        super(S2S_lstm, self).__init__()
        #####################################################################################
        self.vocab_size         = vocab_size
        self.embedding_dim      = embedding_dim
        self.hidden_dim         = hidden_dim
        self.src_pad_token      = src_pad_token
        self.trg_pad_token      = trg_pad_token
        #####################################################################################
        self.embed              = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=None)
        self.bi_lstm_src        = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=2, bias=True,
            batch_first=True, dropout=0.1, bidirectional=True
        ).to(device)
        self.bi_lstm_trg        = nn.LSTM(
            input_size  = 2*self.hidden_dim+self.embedding_dim, hidden_size=self.hidden_dim, num_layers=2, bias=True,
            batch_first = True, dropout=0.1, bidirectional=False
        ).to(device)
        self.projection         = nn.Linear(self.hidden_dim, self.embedding_dim)
        #####################################################################################
        self.loss_f             = SGNS(self.embed, vocab_size=vocab_size, n_negs=20).to(device)
        #####################################################################################
    def forward(self, src_tokens, trg_tokens):
        src_tokens, trg_tokens      = src_tokens.to(device), trg_tokens.to(device)
        # print(src_tokens.size())
        # print(trg_tokens.size())
        src_embeds                  = self.embed(src_tokens)
        trg_embeds                  = self.embed(trg_tokens)
        # print(src_embeds.size())
        src_contextual, (h_n, c_n)  = self.bi_lstm_src(src_embeds)
        # print(src_contextual.size())
        # print(h_n.size())
        # print(c_n.size())
        hidden_concat               = torch.cat((h_n[0], h_n[1]), dim=1).unsqueeze(1).expand(size=(-1, trg_embeds.size(1)-1,-1))
        # print(hidden_concat.size())
        trg_input                   = torch.cat((hidden_concat, trg_embeds[:,:-1,:]), dim=-1)
        # print(trg_input.size())
        trg_contextual, (h_n, c_n)  = self.bi_lstm_trg(trg_input)
        # print(trg_contextual.size())
        out_vecs                    = self.projection(trg_contextual)
        #################################################
        maska                       = (trg_tokens == self.trg_pad_token).float()
        maska                       = (maska-1).abs().unsqueeze(-1)[:,1:].expand_as(out_vecs)
        o1                          = (maska * trg_embeds[:, 1:, :]).reshape(-1, 1, self.embedding_dim)
        o2                          = (maska * out_vecs).reshape(-1, 1, self.embedding_dim)
        #################################################
        loss_                       = self.loss_f(o1, o2)
        # print(loss_)
        return loss_

######################################################################################################
data_path = 'C:\\Users\\dvpap\\Downloads\\quora_duplicate_questions.tsv'
# data_path   = '/home/dpappas/quora_duplicate_questions.tsv'

data_handler    = DataHandler(data_path)
data_handler.save_model('datahandler_model.p')
data_handler.load_model('datahandler_model.p')

exit()

b_size          = 64
vocab_size      = data_handler.vocab_size
SRC_PAD_TOKEN   = data_handler.pad_index
TRGT_PAD_TOKEN  = data_handler.pad_index
embedding_dim   = 30
hidden_dim      = 100
timesteps       = 50
N_EPOCHS        = 10
CLIP            = 1
######################################################################################################
model           = S2S_lstm(
    vocab_size = vocab_size, embedding_dim=embedding_dim, hidden_dim = hidden_dim,
    src_pad_token=SRC_PAD_TOKEN, trg_pad_token=TRGT_PAD_TOKEN
).to(device)
optimizer       = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
######################################################################################################

best_valid_loss = float('inf')
print('TRAINING the model')
for epoch in tqdm(range(N_EPOCHS)):
    start_time  = time.time()
    train_loss  = train_one(clip=CLIP)
    valid_loss  = eval_one()
    end_time    = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'my_s2s_lstms.pt')

# for i in range(1000):
#     loss = model(src_tokens, trg_tokens)
#     print(loss)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()

'''
python -m spacy download en
'''

'''
import pandas as pd
from torchtext import data
from sklearn.model_selection import train_test_split
from torchtext.data import Field, BucketIterator, TabularDataset


to_text, from_text  = [], []
with open(data_path, 'rt', encoding='utf8') as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t')
    for row in tqdm(tsvin, total=404291):
        text1 = row[3]
        text2 = row[4]
        if(not text1):
            continue
        if(not text2):
            continue
        from_text.append(text1)
        from_text.append(text2)
        to_text.append(text2)
        to_text.append(text1)
######################################################################################################
EN_TEXT_1 = Field(tokenize=tokenize_en, init_token = "<sos>", eos_token = "<eos>")
EN_TEXT_2 = Field(tokenize=tokenize_en, init_token = "<sos>", eos_token = "<eos>")
######################################################################################################
raw_data = {'src' : [line for line in from_text], 'trg': [line for line in to_text]}
df = pd.DataFrame(raw_data, columns=["src", "trg"])
df['eng_len'] = df['src'].str.count(' ')
df['fr_len'] = df['trg'].str.count(' ')
df = df.query('fr_len < 80 & eng_len < 80')
df = df.query('fr_len < eng_len * 1.5 & fr_len * 1.5 > eng_len')
######################################################################################################
print('create train and validation set and save to csv')
train_part, val_part = train_test_split(df, test_size=0.1)
train_part.to_csv("train.csv", index=False)
val_part.to_csv("val.csv", index=False)
######################################################################################################
print('Reload to train the model')
data_fields             = [('src', EN_TEXT_1), ('trg', EN_TEXT_2)]
train_part, val_part    = data.TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields)
print('Building vocab')
EN_TEXT_1.build_vocab(train_part, val_part)
EN_TEXT_2.build_vocab(train_part, val_part)
######################################################################################################
SRC_PAD_TOKEN   = EN_TEXT_1.vocab.stoi[EN_TEXT_2.pad_token]
TRGT_PAD_TOKEN  = EN_TEXT_2.vocab.stoi[EN_TEXT_2.pad_token]
######################################################################################################
train_iter      = BucketIterator(train_part, batch_size=b_size, sort_key=lambda x: len(x.trg), shuffle=True)
valid_iter      = BucketIterator(val_part,   batch_size=b_size, sort_key=lambda x: len(x.trg), shuffle=True)
######################################################################################################
'''



