from sklearn.decomposition import PCA
from pprint import pprint
import numpy as np
import pickle
import os
from tqdm import tqdm

diri = '/home/dpappas/bioasq_all/bert_elmo_embeds/'
mat, m = None, 0

for f in tqdm(os.listdir(diri)):
    m += 1
    fpath = os.path.join(diri, f)
    d = pickle.load(open(fpath, 'rb'))
    if (mat is None):
        mat = np.concatenate(d['title_bert_average_embeds'] + d['abs_bert_average_embeds'], axis=0)
    else:
        mat = np.concatenate([mat] + d['title_bert_average_embeds'] + d['abs_bert_average_embeds'], axis=0)
    if (mat.shape[0] > 10000000):
        break

print(m)
print(mat.shape)

# selector    = PCA(n_components=0.9)
selector = PCA(n_components=50)
selector.fit(mat)
temp = selector.transform(mat)
print(temp.shape)

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
