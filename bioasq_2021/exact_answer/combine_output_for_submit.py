
import os, json

# f1      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system1\\batch1_system_1_factoid.json'
# f2      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system1\\System1-Test1.json'
# opath   = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system1\\batch1_submit_system_1.json'

# f1      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system2\\batch1_system_2_factoid.json'
# f2      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system2\\System2-Test1.json'
# opath   = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system2\\batch1_submit_system_2.json'

# f1      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system3\\batch1_system_3_factoid.json'
# f2      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system3\\System3-Test1.json'
# opath   = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system3\\batch1_submit_system_3.json'

# f1      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system4\\batch1_system_4_factoid.json'
# f2      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system4\\BioASQ-Test1.json'
# opath   = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\submit_batch_1\\system4\\batch1_submit_system_4.json'

sys_no  = 5
b       = 4
f1      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\BATCH{}\\PHASE_B\\factoid_results\\batch{}_system_{}_factoid.json'.format(b, b, sys_no)
f2      = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\BATCH{}\\PHASE_B\\summary_results\\System{}-Test{}.json'.format(b, sys_no, b)
opath   = 'C:\\Users\\dvpap\\OneDrive\\Desktop\\bioasq_2021\\BATCH{}\\PHASE_B\\batch{}_submit_system_{}.json'.format(b, b, sys_no)

d1      = json.load(open(f1))
d2      = json.load(open(f2))

d2      = dict((item['id'],item['ideal_answer'].strip()) for item in d2["questions"])

for quest in d1['questions']:
    quest['ideal_answer'] = d2[quest['id']]
    if(quest['type'] == 'yesno'):
        quest['exact_answer'] = 'yes'
    if(quest['type'] == 'factoid'):
        if('exact_answer' in quest ):
            quest['exact_answer'] = [[t] for t in quest['exact_answer'][:5]]
            # quest['exact_answer'] = quest['exact_answer'][:5]

with open(opath, 'w') as of:
    of.write(json.dumps(d1, indent=4, sort_keys=True))
    of.close()

