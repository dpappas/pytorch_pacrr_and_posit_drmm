import os, json

names = [
'bert_bcnn.json',
# 'bertHC_pdrmm.json',
# 'bert_high_bcnn.json',
'bert_pdrmm.json',
'pdrmm_bcnn.json',
'pdrmm_pdrmm.json',
# 'term_pacrr_bcnn.json'
]

odir = '/home/dpappas/bioasq_all/bioasq7/snippet_results/b12345_joined/'
if not os.path.exists(odir):
    os.makedirs(odir)

all_data = {'questions':[]}

for name in names:
    all_data = {'questions':[]}
    for b in range(1, 6):
        data = json.load(open('/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_{}/{}'.format(b, name)))['questions']
        all_data['questions'].extend(data)
    with open(os.path.join(odir, name), 'w') as f:
        f.write(json.dumps(all_data, indent=4, sort_keys=True))
        f.close()

##################################################################################################

names = [
# 'bert-high-conf-0.01.json',
'bert_jpdrmm.json',
# 'bert.json',
# 'JBERT_F.json',
# 'JBERT.json',
# 'jpdrmm.json',
# 'pdrmm.json',
# 'term-pacrr.json'
]

odir = '/home/dpappas/bioasq_all/bioasq7/document_results/b12345_joined/'
if not os.path.exists(odir):
    os.makedirs(odir)

for name in names:
    all_data = {'questions':[]}
    for b in range(1, 6):
        data = json.load(open('/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_{}/{}'.format(b, name)))['questions']
        all_data['questions'].extend(data)
    with open(os.path.join(odir, name), 'w') as f:
        f.write(json.dumps(all_data, indent=4, sort_keys=True))
        f.close()

##################################################################################################

all_data = {'questions':[]}
for b in range(1, 6):
    data = json.load(
        open('/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(b, b))
    )['questions']
    all_data['questions'].extend(data)

with open(os.path.join('/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset12345'), 'w') as f:
    f.write(json.dumps(all_data, indent=4, sort_keys=True))
    f.close()

##################################################################################################
