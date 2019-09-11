

import json, os, copy

diri = '/home/dpappas/bioasq_jpdrmm_2L_0p01_run_0/'
# diri    = '/media/dpappas/dpappas_data/models_out/frozen_bioasq7_JBERT_2L_0p01_run_0/'
# diri    = '/media/dpappas/dpappas_data/models_out/bioasq7_JBERT_2L_0p01_run_0/'
ddd     = json.load(open(os.path.join(diri, 'v3 dev_emit_bioasq.json')))

data_at_3 = {'questions': []}
data_at_5 = {'questions': []}

for q in ddd['questions']:
    d3 = copy.deepcopy(q)
    d5 = copy.deepcopy(q)
    d3['documents'] = d3['documents'][:3]
    d5['documents'] = d5['documents'][:5]
    d3['snippets']  = d3['snippets'][:3]
    d5['snippets']  = d5['snippets'][:5]
    data_at_3['questions'].append(d3)
    data_at_5['questions'].append(d5)

with open(os.path.join(diri, 'v3 dev_emit_bioasq_AT3.json'), 'w') as f:
    f.write(json.dumps(data_at_3, indent=4, sort_keys=False))

with open(os.path.join(diri, 'v3 dev_emit_bioasq_AT5.json'), 'w') as f:
    f.write(json.dumps(data_at_5, indent=4, sort_keys=False))

'''


java -Xmx10G -cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
"/media/dpappas/dpappas_data/models_out/bioasq7_JBERT_2L_0p01_run_0/v3 dev_gold_bioasq.json" \
"/media/dpappas/dpappas_data/models_out/bioasq7_JBERT_2L_0p01_run_0/v3 dev_emit_bioasq_AT3.json"

java -Xmx10G -cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
"/media/dpappas/dpappas_data/models_out/bioasq7_JBERT_2L_0p01_run_0/v3 dev_gold_bioasq.json" \
"/media/dpappas/dpappas_data/models_out/bioasq7_JBERT_2L_0p01_run_0/v3 dev_emit_bioasq_AT5.json"

java -Xmx10G -cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
"/media/dpappas/dpappas_data/models_out/bioasq7_JBERT_2L_0p01_run_0/v3 dev_gold_bioasq.json" \
"/media/dpappas/dpappas_data/models_out/bioasq7_JBERT_2L_0p01_run_0/v3 dev_emit_bioasq_AT5.json"

------------------------------------------------------------------------------------

java -Xmx10G -cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
"/media/dpappas/dpappas_data/models_out/frozen_bioasq7_JBERT_2L_0p01_run_0/v3 dev_gold_bioasq.json" \
"/media/dpappas/dpappas_data/models_out/frozen_bioasq7_JBERT_2L_0p01_run_0/v3 dev_emit_bioasq_AT3.json"

java -Xmx10G -cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
"/media/dpappas/dpappas_data/models_out/frozen_bioasq7_JBERT_2L_0p01_run_0/v3 dev_gold_bioasq.json" \
"/media/dpappas/dpappas_data/models_out/frozen_bioasq7_JBERT_2L_0p01_run_0/v3 dev_emit_bioasq_AT5.json"

java -Xmx10G -cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
"/media/dpappas/dpappas_data/models_out/frozen_bioasq7_JBERT_2L_0p01_run_0/v3 dev_gold_bioasq.json" \
"/media/dpappas/dpappas_data/models_out/frozen_bioasq7_JBERT_2L_0p01_run_0/v3 dev_emit_bioasq.json"

------------------------------------------------------------------------------------



java -Xmx10G -cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
"/home/dpappas/bioasq_jpdrmm_2L_0p01_run_0/v3 dev_gold_bioasq.json" \
"/home/dpappas/bioasq_jpdrmm_2L_0p01_run_0/v3 dev_emit_bioasq_AT3.json"

java -Xmx10G -cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
"/home/dpappas/bioasq_jpdrmm_2L_0p01_run_0/v3 dev_gold_bioasq.json" \
"/home/dpappas/bioasq_jpdrmm_2L_0p01_run_0/v3 dev_emit_bioasq_AT5.json"

java -Xmx10G -cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA -e 5  \
"/home/dpappas/bioasq_jpdrmm_2L_0p01_run_0/v3 dev_gold_bioasq.json" \
"/home/dpappas/bioasq_jpdrmm_2L_0p01_run_0/v3 dev_emit_bioasq.json"

------------------------------------------------------------------------------------


'''














