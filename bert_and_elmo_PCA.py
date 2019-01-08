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
        mat = np.concatenate(d['title_bert_average_embeds'] + d['abs_bert_average_embeds'], axis=0)
    else:
        mat = np.concatenate([mat] + d['title_bert_average_embeds'] + d['abs_bert_average_embeds'], axis=0)
    if (mat.shape[0] > 1000):
        transformer.partial_fit(
            np.concatenate(d['title_bert_average_embeds'] + d['abs_bert_average_embeds'], axis=0)
        )
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
