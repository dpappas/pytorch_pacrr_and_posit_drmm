

import json
from pprint import pprint

fpath = '/home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/drmm-experimental_submit.json'
opath = '/home/dpappas/drmm-experimental_submit.json'

data  = json.load(open(fpath))

for quest in data['questions']:
    for sn in quest['snippets']:
        del(sn['offsetInBeginSection'])
        del(sn['offsetInEndSection'])
        del(sn['beginSection'])
        del(sn['endSection'])
    # pprint(quest)

with open(opath, 'w') as f:
    f.write(json.dumps(data, indent=4, sort_keys=True))











