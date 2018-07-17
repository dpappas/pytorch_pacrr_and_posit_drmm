

import cPickle as pickle
import numpy as np

token_to_index_f    = '/home/dpappas/joint_task_list_batches/t2i.p'
t2i                 = pickle.load(open(token_to_index_f,'rb'))
idf                 = pickle.load(open('/home/dpappas/IDF_python_v2.pkl', 'rb'))
idf_mat             = np.zeros(len(t2i))
max_idf             = max(idf.keys())
for t in t2i:
    try:
        idf_mat[t2i[t]] = idf[t]
    except KeyError:
        idf_mat[t2i[t]] = max_idf

np.save('/home/dpappas/joint_task_list_batches/idf_matrix.npy', idf_mat)
