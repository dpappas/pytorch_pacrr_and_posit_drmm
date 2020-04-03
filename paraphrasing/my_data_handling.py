
import csv, re
from tqdm import tqdm
from collections import Counter

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())

class DataHandler:
   def __init__(self, data_path= '/home/dpappas/quora_duplicate_questions.tsv', occur_thresh=3):
       self.to_text, self.from_text = [], []
       self.occur_thresh            = occur_thresh
       self.vocab                   = Counter()
       with open(data_path, 'rt', encoding='utf8') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tqdm(tsvin, total=404291, desc='Reading file'):
                ################################################
                text1 = bioclean(row[3])
                text2 = bioclean(row[4])
                ################################################
                if(not text1):
                    continue
                if(not text2):
                    continue
                ################################################
                self.vocab.update(Counter(text1))
                self.vocab.update(Counter(text2))
                ################################################
                self.from_text.append(text1)
                self.from_text.append(text2)
                self.to_text.append(text2)
                self.to_text.append(text1)
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






