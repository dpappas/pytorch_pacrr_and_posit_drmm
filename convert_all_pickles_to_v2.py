
import pickle
import os
from tqdm import tqdm

diri    = '/home/dpappas/bioasq_all/bert_elmo_embeds/'
pbar    = tqdm(os.listdir(diri))
for f in pbar:
    pbar.set_description(f)
    d = pickle.load(open(os.path.join(diri, f), 'rb'))
    pickle.dump(d, open(os.path.join(diri, f), 'wb'), protocol=2)







