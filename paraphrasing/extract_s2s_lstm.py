
import torch, re, random, spacy, time, pickle, os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
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
        oloss       = (oloss.sigmoid()+1e-05).log()
        oloss       = oloss.mean(1)
        nloss       = torch.bmm(nvectors, true_vecs.transpose(1,2))
        nloss       = (nloss.squeeze().sigmoid()+1e-05).log()
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
        self.out_layer          = nn.Linear(self.hidden_dim, self.vocab_size)
        #####################################################################################
        # self.loss_f             = SGNS(self.embed, vocab_size=vocab_size, n_negs=40).to(device)
        self.loss_f             = nn.CrossEntropyLoss()
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
        out_vecs                    = self.out_layer(trg_contextual)
        # # print(out_vecs.size())
        # # return
        # out_vecs                    = F.sigmoid(out_vecs)
        # #################################################
        # maska                       = (trg_tokens == self.trg_pad_token).float()
        # maska                       = (maska-1).abs().unsqueeze(-1)[:,1:].expand_as(out_vecs)
        # o1                          = (maska * trg_embeds[:, 1:, :]).reshape(-1, 1, self.embedding_dim)
        # o2                          = (maska * out_vecs).reshape(-1, 1, self.embedding_dim)
        # #################################################
        # loss_                       = self.loss_f(o1, o2)
        # # print(loss_)
        # print(out_vecs.size())
        # print(trg_tokens.size())
        loss_                       = self.loss_f(out_vecs.reshape(-1, out_vecs.size(2)), trg_tokens[:,1:].reshape(-1))
        return loss_

#############################################################
# data_path = 'C:\\Users\\dvpap\\Downloads\\quora_duplicate_questions.tsv'
data_path   = '/home/dpappas/quora_duplicate_questions.tsv'

data_handler    = DataHandler(data_path)
data_handler.load_model('datahandler_model.p')

b_size          = 64
vocab_size      = data_handler.vocab_size
SRC_PAD_TOKEN   = data_handler.pad_index
TRGT_PAD_TOKEN  = data_handler.pad_index
embedding_dim   = 30
hidden_dim      = 50
N_EPOCHS        = 10
CLIP            = 1
######################################################################################################
model           = S2S_lstm(
    vocab_size = vocab_size, embedding_dim=embedding_dim, hidden_dim = hidden_dim,
    src_pad_token=SRC_PAD_TOKEN, trg_pad_token=TRGT_PAD_TOKEN
).to(device)
######################################################################################################

def load_model_from_checkpoint(resume_dir):
    global start_epoch, optimizer
    resume_from = os.path.join(resume_dir, 'my_s2s_lstms.pt')
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        #############################################################################################
        model.load_state_dict(checkpoint)
        #############################################################################################
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

resume_dir = './'
load_model_from_checkpoint(resume_dir)

model.eval()

# pprint(data_handler.train_instances[:10])

question    = 'what is an ethical dilemma ?'
src_tokens  = data_handler.encode_one(question)
src_tokens  = torch.LongTensor(src_tokens).to(device)
############################################################
src_embeds  = model.embed(src_tokens)
src_contextual, (h_n, c_n) = model.bi_lstm_src(src_embeds)
############################################################
h_0             = None #  (num_layers * num_directions, batch, hidden_size)
c_0             = None #  (num_layers * num_directions, batch, hidden_size)
next_token      = '<EOS>'
all_tokens      = [next_token]
while(True):
    c1              = torch.cat((h_n[0], h_n[1]), dim=1).unsqueeze(1)
    c2              = model.embed(torch.LongTensor(data_handler.encode_one(next_token)).to(device))
    ############################################################
    if(h_0 is None):
        trg_input                   = torch.cat((c1, c2), dim=-1)
        trg_contextual, (h_0, c_0)  = model.bi_lstm_trg(trg_input)
        # next_token                  = data_handler.itos[model.out_layer(trg_contextual).argmax().item()]
        ############################################################
        tt                          = model.out_layer(trg_contextual).sort().indices.squeeze()
        j                           = len(tt)-1
        while True:
            if(tt[j] != 1):
                break
            j -= 1
        next_token                  = data_handler.itos[tt[j].item()]
        ############################################################
    else:
        trg_input                   = torch.cat((c1, c2), dim=-1)
        trg_contextual, (h_0, c_0)  = model.bi_lstm_trg(trg_input, (h_0, c_0))
        # next_token                  = data_handler.itos[model.out_layer(trg_contextual).argmax().item()]
        ############################################################
        tt                          = model.out_layer(trg_contextual).sort().indices.squeeze()
        j                           = len(tt)-1
        while True:
            if(tt[j] != 1):
                break
            j -= 1
        next_token                  = data_handler.itos[tt[j].item()]
        ############################################################
    all_tokens.append(next_token)
    if(len(all_tokens) == 10 or next_token == '<EOS>'):
        break

print(all_tokens)

############################################################

