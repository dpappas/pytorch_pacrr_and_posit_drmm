
from pprint import pprint
import json

fpaths = [
    (
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\exact_answer\\system_1.json',
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\system_1.json'
    ),
    (
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\exact_answer\\system_5.json',
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\system_5.json'
    )
]

fff = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\BioASQ-task8bPhaseB-testset2.json'
d2  = json.load(open(fff))
tt  = dict((q['id'], q['type']) for q in d2['questions'])
# pprint(tt)

for f2, opath in fpaths:
    d2 = json.load(open(f2))
    for quest in d2['questions']:
        quest['type']   = tt[quest['id']]
        #######################################
        if(quest['type'] == 'yesno'):
            quest['exact_answer'] = 'yes'
        if(quest['type'] == 'summary'):
            quest['ideal_answer'] = [' '.join(s['text'] for s in quest['snippets'])]
        #######################################
    with open(opath, 'w') as of:
        of.write(json.dumps(d2, indent=4, sort_keys=True))
        of.close()

