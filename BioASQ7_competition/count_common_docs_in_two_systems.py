

import json

fpath1  = '/home/DATA/Biomedical/bioasq7/submit_files/test_batch_1/batch1-sys1.json'
fpath2  = '/home/DATA/Biomedical/bioasq7/submit_files/test_batch_1/batch1-sys2.json'

d1      = json.load(open(fpath1))
d2      = json.load(open(fpath2))

# common  =
# total   =


docs1 = d1['questions'][0]['documents']
docs2 = d2['questions'][0]['documents']

# set(docs1) - set(docs2)
len(set(docs1).intersection(set(docs2)))



