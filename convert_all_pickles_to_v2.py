
import pickle
import os
from tqdm import tqdm

diri = '/home/dpappas/bioasq_all/bert_elmo_embeds/'

for f in tqdm(os.listdir(diri)):
    d = pickle.load(open(os.path.join(diri, f), 'rb'))
    pickle.dump(d, open(os.path.join(diri, f), 'wb'), protocol=2)







