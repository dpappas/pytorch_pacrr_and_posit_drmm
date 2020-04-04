
import csv, re
from tqdm import tqdm
from collections import Counter

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())

class DataHandler:
   def __init__(self, data_path= '/home/dpappas/quora_duplicate_questions.tsv', occur_thresh=5, valid_split=0.1):
       self.to_text, self.from_text = [], []
       self.occur_thresh            = occur_thresh
       ################################################
       with open(data_path, 'rt', encoding='utf8') as tsvin:
           tsvin = csv.reader(tsvin, delimiter='\t')
           headers = next(tsvin)
           for row in tqdm(tsvin, total=404291, desc='Reading file'):
               ################################################
               text1 = bioclean(row[3])
               text2 = bioclean(row[4])
               ################################################
               if(not text1):
                   continue
               if(not text2):
                   continue
               if(len(text1.split())<=3):
                   continue
               if(len(text2.split())<=3):
                   continue
               if(text1 == text2[:len(text1)]):
                   continue
               if(text2 == text1[:len(text2)]):
                   continue
               if(all(t in set(text2.split()) for t in set(text1.split()))):
                   continue
               ################################################
               self.from_text.append(text1)
               self.from_text.append(text2)
               self.to_text.append(text2)
               self.to_text.append(text1)
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
       train_instances  = sorted(list(zip(self.train_from_text, self.train_to_text)), key= lambda x: len(x[0].split())*10000+len(x[1].split()))
       dev_instances    = sorted(list(zip(self.dev_from_text, self.dev_to_text)),     key= lambda x: len(x[0].split())*10000+len(x[1].split()))
       print(dev_instances[0])
       print(train_instances[0])
       ################################################
       self.vocab           = Counter()
       self.vocab.update(Counter(' '.join(self.train_from_text).split()))
       self.vocab.update(Counter(' '.join(self.train_to_text).split()))
       ################################################
       self.vocab                   = sorted(
           [
               word
               for word in tqdm(self.vocab, desc='Building VOCAB')
               if(self.vocab[word]>=self.occur_thresh)
           ],
           key= lambda x: self.vocab[x]
       )
       self.vocab                   = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + self.vocab
       self.stoi                    = dict(enumerate(self.vocab))
       self.itos                    = dict((v, k) for k,v in self.stoi.items())
       print('Kept {} total words'.format(len(self.vocab)))
   def displayEmployee(self):
      print("Name : ", self.name,  ", Salary: ", self.salary)






