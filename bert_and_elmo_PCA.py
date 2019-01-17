# from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from pprint import pprint
import numpy as np
import pickle
import os
from tqdm import tqdm

diri = '/home/dpappas/bioasq_all/bert_elmo_embeds/'
mat, m = None, 0

transformer = IncrementalPCA(n_components=50)
for f in tqdm(os.listdir(diri), ascii=True):
    m += 1
    fpath = os.path.join(diri, f)
    d = pickle.load(open(fpath, 'rb'))
    #
    if (mat is None):
        mat = np.concatenate(d['title_sent_elmo_embeds'] + d['abs_sent_elmo_embeds'], axis=0)
    else:
        mat = np.concatenate([mat] + d['title_sent_elmo_embeds'] + d['abs_sent_elmo_embeds'], axis=0)
    if (mat.shape[0] > 1000):
        transformer.partial_fit(mat)
        mat = None

filename = '/home/dpappas/bioasq_all/pca_elmo_transformer.sav'
pickle.dump(transformer, open(filename, 'wb'))

#####################

import os
import pickle
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

diri = '/home/dpappas/bioasq_all/bert_elmo_embeds/'
filename = '/home/dpappas/bioasq_all/pca_bert_transformer.sav'
transformer = pickle.load(open(filename, 'rb'))

odiri = '/home/dpappas/bioasq_all/bert_embeds_after_pca/'
if (not os.path.exists(odiri)):
    os.makedirs(odiri)

for f in tqdm(os.listdir(diri), ascii=True):
    fpath = os.path.join(diri, f)
    opath = os.path.join(odiri, f)
    if (not os.path.exists(opath)):
        d = pickle.load(open(fpath, 'rb'))
        od = {
            'title_elmo_average_embeds': [transformer.transform(m) for m in d['title_sent_elmo_embeds']],
            'abs_elmo_average_embeds': [transformer.transform(m) for m in d['abs_sent_elmo_embeds']],
            'mesh_elmo_average_embeds': [transformer.transform(m) for m in d['mesh_elmo_embeds']],
        }
        pickle.dump(od, open(opath, 'wb'), protocol=2)

all_qs = pickle.load(open('/home/dpappas/bioasq_all/all_quest_embeds.p', 'rb'))
all_qs_pca = {}

for quest in tqdm(all_qs.keys(), ascii=True):
    all_qs_pca[quest] = [transformer.transform(m) for m in all_qs[quest]['title_sent_elmo_embeds']]

pickle.dump(all_qs_pca, open('/home/dpappas/bioasq_all/all_quest_elmo_embeds_after_pca.p', 'wb'), protocol=2)

exit()

transformer = IncrementalPCA(n_components=50)
for f in tqdm(os.listdir(diri), ascii=True):
    m += 1
    fpath = os.path.join(diri, f)
    d = pickle.load(open(fpath, 'rb'))
    #
    if (mat is None):
        mat = np.concatenate(d['title_bert_average_embeds'] + d['abs_bert_average_embeds'], axis=0)
    else:
        mat = np.concatenate([mat] + d['title_bert_average_embeds'] + d['abs_bert_average_embeds'], axis=0)
    if (mat.shape[0] > 1000):
        transformer.partial_fit(mat)
        mat = None

filename = '/home/dpappas/bioasq_all/pca_bert_transformer.sav'
pickle.dump(transformer, open(filename, 'wb'))

'''
# [
# 'title_sent_elmo_embeds',
# 'abs_sent_elmo_embeds',
# 'mesh_elmo_embeds',
# 'title_bert_average_embeds',
# 'title_bert_original_embeds',
# 'title_bert_original_tokens',
# 'abs_bert_average_embeds',
# 'abs_bert_original_embeds',
# 'abs_bert_original_tokens',
# 'mesh_bert_average_embeds',
# 'mesh_bert_original_embeds',
# 'mesh_bert_original_tokens'
# ]
'''

import os
import pickle
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

diri = '/home/dpappas/bioasq_all/bert_elmo_embeds/'
filename = '/home/dpappas/bioasq_all/pca_bert_transformer.sav'
transformer = pickle.load(open(filename, 'rb'))

odiri = '/home/dpappas/bioasq_all/bert_embeds_after_pca/'
if (not os.path.exists(odiri)):
    os.makedirs(odiri)

for f in tqdm(os.listdir(diri), ascii=True):
    fpath = os.path.join(diri, f)
    opath = os.path.join(odiri, f)
    if (not os.path.exists(opath)):
        d = pickle.load(open(fpath, 'rb'))
        od = {
            'title_bert_average_embeds': [transformer.transform(m) for m in d['title_bert_average_embeds']],
            'abs_bert_average_embeds': [transformer.transform(m) for m in d['abs_bert_average_embeds']],
            'mesh_bert_average_embeds': [transformer.transform(m) for m in d['mesh_bert_average_embeds']],
        }
        pickle.dump(od, open(opath, 'wb'), protocol=2)

all_qs = pickle.load(open('/home/dpappas/bioasq_all/all_quest_embeds.p', 'rb'))
all_qs_pca = {}

for quest in tqdm(all_qs.keys(), ascii=True):
    all_qs_pca[quest] = [transformer.transform(m) for m in all_qs[quest]['title_bert_average_embeds']]

pickle.dump(all_qs_pca, open('/home/dpappas/bioasq_all/all_quest_bert_embeds_after_pca.p', 'wb'), protocol=2)

python3
.6
import pickle

d = pickle.load(open('/home/dpappas/bioasq_all/bert_elmo_embeds/1002140.p', 'rb'))
