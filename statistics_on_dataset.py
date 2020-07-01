
# Train
# total instances: 2647
#
# Dev
# total instances: 100
#
#
# Test
# total instances: 100


from nltk import sent_tokenize
from tqdm import tqdm
import pickle, re

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

dataloc     = '/home/dpappas/bioasq_all/bioasq7_data/'

with open(dataloc + 'bioasq7_bm25_docset_top100.train.pkl', 'rb') as f:
    train_docs = pickle.load(f)

lenss = []
lenss2 = []
lenss_tit = []
lenss_abs = []
lenss_tit2 = []
lenss_abs2 = []
for d in tqdm(train_docs.values()):
    lenss.append(len(bioclean(d['title'])) + len(bioclean(d['abstractText'])))
    lenss.append(len(d['title'].split()) + len(d['abstractText'].split()))
    lenss_tit.append(len(bioclean(d['title'])))
    lenss_abs.append(len(bioclean(d['abstractText'])))
    lenss_tit2.append(len(d['title'].split()))
    lenss_abs2.append(len(d['abstractText'].split()))

import numpy as np

np.max(lenss)
np.min(lenss)
np.average(lenss)

np.max(lenss2)
np.min(lenss2)
np.average(lenss2)


# length of titles
# max: 73
# min: 1
# average: 12.723450985111578
#
#
# length of asbtracts
# max: 1494
# min: 1
# average: 184.46201389127668


import json
with open(dataloc+'trainining7b.json', 'r') as f:
	bioasq7_data = json.load(f)
	bioasq7_data = dict((q['id'], q) for q in bioasq7_data['questions'])

lenss_q  = []
lenss_q2 = []
for item in bioasq7_data.values():
    lenss_q.append(len(bioclean(item['body'])))
    lenss_q2.append(len(item['body'].split()))


np.max(lenss_q)
np.min(lenss_q)
np.average(lenss_q)

np.max(lenss_q2)
np.min(lenss_q2)
np.average(lenss_q2)



# length of questions bioclean
# max: 30
# min: 2
# average: 9.005096468875136
#
#
#
# length of questions whitespace
# max: 30
# min: 2
# average: 9.016745540589735


