
import json

fpaths = [
    (
        'C:/Users/dvpap/OneDrive/Desktop/bioasq8_submitted/bioasq8_batch1_system1_results.json',
        'C:/Users/dvpap/OneDrive/Desktop/bioasq8_submitted factoid/system_1_jpdrmm_sgrank.json'
    ),
    (
        'C:/Users/dvpap/OneDrive/Desktop/bioasq8_submitted/bioasq8_batch1_system3_results.json',
        'C:/Users/dvpap/OneDrive/Desktop/bioasq8_submitted factoid/system_3_jpdrmm_bertreader.json'
    ),
    (
        'C:/Users/dvpap/OneDrive/Desktop/bioasq8_submitted/bioasq8_batch1_system4_results.json',
        'C:/Users/dvpap/OneDrive/Desktop/bioasq8_submitted factoid/system_4_gold_bertreader.json'
    ),
    (
        'C:/Users/dvpap/OneDrive/Desktop/bioasq8_submitted/bioasq8_batch1_system5_results.json',
        'C:/Users/dvpap/OneDrive/Desktop/bioasq8_submitted factoid/system_5_gold_sgrank.json'
    )
]

c = 0
for f1, f2 in fpaths:
    c += 1
    #####################################################################
    d2 = json.load(open(f2))
    tt = dict((q['id'], (q['type'], q['exact_answer'] if 'exact_answer' in q else None))for q in d2['questions'])
    #####################################################################
    d1 = json.load(open(f1))
    for quest in d1['questions']:
        if(quest['id'] in tt):
            quest['type'] = tt[quest['id']][0]
            if(tt[quest['id']][1] is not None):
                quest['exact_answer'] = tt[quest['id']][1]
            if(quest['type'] == 'yesno'):
                quest['exact_answer'] = 'yes'
            if(quest['type'] == 'summary'):
                quest['ideal_answer'] = [' '.join(s['text'] for s in quest['snippets'])]
    #####################################################################
    with open('C:/Users/dvpap/OneDrive/Desktop/system_{}.json'.format(c), 'w') as of:
        of.write(json.dumps(d1, indent=4, sort_keys=True))
        of.close()

