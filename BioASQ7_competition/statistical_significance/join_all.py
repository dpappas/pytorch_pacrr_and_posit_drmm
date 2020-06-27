import os, json
from pprint import pprint

# ddd = "/media/dpappas/dpappas_data/models_out/"
# basedirs = [os.path.join(ddd, sd) for sd in os.listdir(ddd)]
# basedirs.append('/home/dpappas/ablation_1111111_0p01_0_bioasq_jpdrmm_2L_0p01_run_0/')

basedirs = [
    # '/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_NORESCORE_adapt_frozen_run_0_WL_1.0_0.0/'
    '/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_maxsentasdoc_adapt_frozen_run_0_WL_0.1_1.0/'
]

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

#


'''

java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_NORESCORE_adapt_frozen_run_0_WL_1.0_0.0/all_res_12345.json" \
| grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/dpappas/ablation_1111111_0p01_0_bioasq_jpdrmm_2L_0p01_run_0/all_res_12345.json" \
| grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar' evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/media/dpappas/dpappas_data/models_out/bioasq7_bertjpdrmadaptnf_maxsentasdoc_adapt_frozen_run_0_WL_0.1_1.0/all_res_12345.json" \
| grep "^MAP documents:\|^MAP snippets:"

'''