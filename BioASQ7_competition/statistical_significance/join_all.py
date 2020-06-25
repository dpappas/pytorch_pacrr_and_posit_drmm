import os, json
from pprint import pprint

ddd = "/media/dpappas/dpappas_data/models_out/"

basedirs = [os.path.join(ddd, sd) for sd in os.listdir(ddd)]

for basedir in basedirs:
    try:
        all_d = {'questions':[]}
        for b in range(1,6):
            fpath = os.path.join(basedir,'batch_{}'.format(b),'v3 test_emit_bioasq.json')
            # print(fpath)
            d = json.load(open(fpath))
            all_d['questions'].extend(d['questions'])
            with open(os.path.join(basedir, 'all_res_12345.json'), 'w') as f:
                gb = f.write(json.dumps(all_d, indent=4, sort_keys=True))
    except:
        print(basedir)

# /home/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345


