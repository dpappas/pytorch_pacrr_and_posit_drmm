
import csv, re, pickle
from tqdm import tqdm
from collections import Counter
import numpy as np

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())

def check_texts(text1, text2):
    if (not text1):
        return False
    if (not text2):
        return False
    if (len(text1.split()) <= 3):
        return False
    if (len(text2.split()) <= 3):
        return False
    if (text1 == text2[:len(text1)]):
        return False
    if (text2 == text1[:len(text2)]):
        return False
    if (all(t in set(text2.split()) for t in set(text1.split()))):
        return False
    return True

class DataHandler:
    def __init__(self, data_path= '/home/dpappas/quora_duplicate_questions.tsv', occur_thresh=5, valid_split=0.1):
        self.to_text, self.from_text = [], []
        self.occur_thresh   = occur_thresh
        self.data_path      = data_path
        ################################################
        with open(data_path, 'rt', encoding='utf8') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            headers = next(tsvin)
            del(headers)
            for row in tqdm(tsvin, total=404291, desc='Reading file'):
                ################################################
                text1 = bioclean(row[3])
                text2 = bioclean(row[4])
                ################################################
                if(not check_texts(text1, text2)):
                    continue
                ################################################
                self.from_text.append(text1)
                self.from_text.append(text2)
                self.to_text.append(text2)
                self.to_text.append(text1)
                ################################################
            print('Created {} examples'.format(len(self.from_text)))
            ################################################
            self.train_from_text = self.from_text[:int(len(self.from_text)*(1. - valid_split))]
            self.train_to_text   = self.to_text[:int(len(self.to_text)*(1. - valid_split))]
            self.dev_from_text   = self.from_text[-int(len(self.from_text)*(valid_split)):]
            self.dev_to_text     = self.to_text[-int(len(self.to_text)*(valid_split)):]
            print('FROM: kept {} instances for training and {} for eval'.format(len(self.train_from_text), len(self.dev_from_text)))
            print('TO: kept {} instances for training and {} for eval'.format(len(self.train_to_text), len(self.dev_to_text)))
            del(self.to_text)
            del(self.from_text)
            ################################################ SORT INSTANCES BY SIZE
            self.train_instances  = sorted(list(zip(self.train_from_text, self.train_to_text)), key= lambda x: len(x[0].split())*10000+len(x[1].split()))
            self.dev_instances    = sorted(list(zip(self.dev_from_text, self.dev_to_text)),     key= lambda x: len(x[0].split())*10000+len(x[1].split()))
            self.number_of_train_instances  = len(self.train_instances)
            self.number_of_dev_instances    = len(self.dev_instances)
            # print('{} instances for training and {} for eval'.format(len(self.train_instances), len(self.dev_instances)))
            # # print(self.dev_instances[0])
            # # print(self.train_instances[0])
            ################################################
            self.vocab                  = Counter()
            self.vocab.update(Counter(' '.join(self.train_from_text).split()))
            self.vocab.update(Counter(' '.join(self.train_to_text).split()))
            ################################################
            self.vocab                  = sorted([
                   word
                   for word in tqdm(self.vocab, desc='Building VOCAB')
                   if(self.vocab[word]>=self.occur_thresh)
               ], key= lambda x: self.vocab[x])
            self.vocab                  = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + self.vocab
            self.itos                   = dict(enumerate(self.vocab))
            self.stoi                   = dict((v, k) for k,v in self.itos.items())
            self.vocab_size             = len(self.vocab)
            print('Kept {} total words'.format(len(self.vocab)))
            ################################################
            self.unk_token              = '<UNK>'
            self.pad_token              = '<PAD>'
            self.unk_index              = self.stoi['<UNK>']
            self.pad_index              = self.stoi['<PAD>']
            ################################################
    def fix_one_batch(self, batch):
        max_len_s = max([len(row) for row in batch['src_ids']])
        max_len_t = max([len(row) for row in batch['trg_ids']])
        batch['src_ids'] = np.array([row + ([self.stoi['<PAD>']] * (max_len_s - len(row))) for row in batch['src_ids']])
        batch['trg_ids'] = np.array([row + ([self.stoi['<PAD>']] * (max_len_t - len(row))) for row in batch['trg_ids']])
        return batch
    def iter_train_batches(self, batch_size):
        self.train_total_batches = int(len(self.train_instances) / batch_size)
        # pbar         = tqdm(total=self.train_total_batches+1)
        batch        = {'src_ids': [], 'trg_ids': []}
        for text_s, text_t in self.train_instances:
            batch['src_ids'].append([self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in text_s.split()])
            batch['trg_ids'].append([self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in text_t.split()])
            if(len(batch['src_ids']) == batch_size):
                # pbar.update(1)
                yield self.fix_one_batch(batch)
                batch = {'src_ids': [], 'trg_ids': []}
        if(len(batch['src_ids'])):
            # pbar.update(1)
            yield self.fix_one_batch(batch)
    def iter_dev_batches(self, batch_size):
        self.dev_total_batches   = int(len(self.dev_instances) / batch_size)
        pbar                     = tqdm(total=self.dev_total_batches+1)
        batch                    = {'src_ids': [], 'trg_ids': []}
        for text_s, text_t in self.dev_instances:
            batch['src_ids'].append([self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in text_s.split()])
            batch['trg_ids'].append([self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in text_t.split()])
            if(len(batch['src_ids']) == batch_size):
                pbar.update(1)
                yield self.fix_one_batch(batch)
                batch = {'src_ids': [], 'trg_ids': []}
        if(len(batch['src_ids'])):
            pbar.update(1)
            yield self.fix_one_batch(batch)
    def save_model(self, pickle_datapath):
        data_handler_data = {
            'stoi'      : self.stoi,
            'vocab'     : self.vocab,
            'data_path' : self.data_path,
            'unk_token' : self.unk_token,
            'pad_token' : self.pad_token
        }
        pickle.dump(data_handler_data, open(pickle_datapath, 'wb'))
    def load_model(self, pickle_datapath):
        data_handler_data   = pickle.load(open(pickle_datapath, 'rb'))
        self.stoi           = data_handler_data['stoi']
        self.itos           = dict((v, k) for k,v in self.stoi.items())
        self.vocab          = data_handler_data['vocab']
        self.data_path      = data_handler_data['data_path']
        self.unk_token      = data_handler_data['unk_token']
        self.pad_token      = data_handler_data['pad_token']




