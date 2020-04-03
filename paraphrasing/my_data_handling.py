
import csv, re
from tqdm import tqdm
from collections import Counter

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())

class DataHandler:
   def __init__(self, data_path= '/home/dpappas/quora_duplicate_questions.tsv'):
       self.to_text, self.from_text = [], []
       self.to_text, self.from_text = [], []
       self.vocab                   = Counter()
       self.vocab['<eos>']          = -1
       self.vocab['<sos>']          = -1
       self.vocab['<pad>']          = -1
       self.vocab['<unk>']          = -1
       with open(data_path, 'rt', encoding='utf8') as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for row in tqdm(tsvin, total=404291):
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
   def displayEmployee(self):
      print("Name : ", self.name,  ", Salary: ", self.salary)






