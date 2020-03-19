
from pprint import pprint
import json

fpaths = [
    (
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\ir_results\\system1.json',
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\exact_answer\\system_3.json',
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\system_3.json'
    ),
    (
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\ir_results\\system2.json',
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\exact_answer\\system_2.json',
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\system_2.json'
    ),
    (
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\BioASQ-task8bPhaseB-testset2.json',
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\exact_answer\\system_4.json',
        'C:\\Users\\dvpap\\OneDrive\\Desktop\\system_4.json'
    )
]

fff = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\batch2_submit_files\\BioASQ-task8bPhaseB-testset2.json'
d2  = json.load(open(fff))
tt  = dict((q['id'], q['type']) for q in d2['questions'])
# pprint(tt)

for f1, f2, opath in fpaths:
    print(f1)
    print(f2)
    d1 = json.load(open(f1))
    d2 = json.load(open(f2))
    # pprint(d2)
    for quest, res in zip(d1['questions'], d2):
        #######################################
        quest['type'] = tt[quest['id']]
        #######################################
        if(quest['type'] == 'factoid'):
            quest['exact_answer'] = [t['text'] for t in res['ngrams']]
        if(quest['type'] == 'list'):
            quest['exact_answer'] = [[t['text']] for t in res['ngrams']]
        if(quest['type'] == 'yesno'):
            quest['exact_answer'] = 'yes'
        if(quest['type'] == 'summary'):
            quest['ideal_answer'] = [' '.join(s['text'] for s in quest['snippets'])]
        #######################################
    with open(opath, 'w') as of:
        of.write(json.dumps(d1, indent=4, sort_keys=True))
        of.close()






