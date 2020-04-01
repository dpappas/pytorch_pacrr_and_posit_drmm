


from tqdm import tqdm
import csv

to_text     = []
from_text   = []
with open('/home/dpappas/seq2seq_paraphrasing/quora_duplicate_questions.tsv', 'rt', encoding='utf8') as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t')
    for row in tqdm(tsvin):
        from_text.append(row[3])
        from_text.append(row[4])
        to_text.append(row[4])
        to_text.append(row[3])

######################################################################################################

import time, math
import torch
import torch.nn as nn
import spacy
import torchtext
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext import data
from torchtext.data import Field, BucketIterator, TabularDataset
en = spacy.load('en')

######################################################################################################

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=150):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

######################################################################################################

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=150):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]
        batch_size  = src.shape[0]
        src_len     = src.shape[1]
        pos         = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src = [batch size, src len, hid dim]
        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [batch size, src len, hid dim]
        return src

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]
        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        # dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, query len, key len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]
        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]
        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=150):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, trg len]
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = [batch size, trg len, hid dim]
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # dropout, residual connection and layer norm
        trg     = self.layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self,encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask
    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # trg_pad_mask = [batch size, 1, trg len, 1]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask
    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        src, trg = src.to(device), trg.to(device)
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch size, src len, hid dim]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        return output, attention

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
# create train and validation set
train_part, val_part = train_test_split(df, test_size=0.1)
train_part.to_csv("train.csv", index=False)
val_part.to_csv("val.csv", index=False)
######################################################################################################
data_fields             = [('src', EN_TEXT_1), ('trg', EN_TEXT_2)]
train_part, val_part    = data.TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields)
EN_TEXT_1.build_vocab(train_part, val_part)
EN_TEXT_2.build_vocab(train_part, val_part)
######################################################################################################
train_iter = BucketIterator(train_part, batch_size=20, sort_key=lambda x: len(x.trg), shuffle=True)
valid_iter = BucketIterator(val_part, batch_size=20, sort_key=lambda x: len(x.trg), shuffle=True)
test_iter  = valid_iter
######################################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################################################################

INPUT_DIM   = len(EN_TEXT_1.vocab)
OUTPUT_DIM  = len(EN_TEXT_2.vocab)
HID_DIM     = 256
ENC_LAYERS  = 3
DEC_LAYERS  = 3
ENC_HEADS   = 8
DEC_HEADS   = 8
ENC_PF_DIM  = 512
DEC_PF_DIM  = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc         = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec         = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
SRC_PAD_IDX = EN_TEXT_1.vocab.stoi[EN_TEXT_1.pad_token]
TRG_PAD_IDX = EN_TEXT_2.vocab.stoi[EN_TEXT_2.pad_token]
model       = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights)
LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

# next(enumerate(train_iter))[1]

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut6-model.pt'))
test_loss = evaluate(model, test_iter, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

example_idx = 8

src = vars(train_part.examples[example_idx])['src']
trg = vars(train_part.examples[example_idx])['trg']

print(f'src = {src}')
print(f'trg = {trg}')
translation, attention = translate_sentence(src, EN_TEXT_1, EN_TEXT_2, model, device)
print(f'predicted trg = {translation}')

'''
CUDA_VISIBLE_DEVICES=-1 python3.6 train_model.py 
'''