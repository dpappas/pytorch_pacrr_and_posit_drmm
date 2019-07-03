#!/usr/bin/env bash


echo "$1"

echo "/media/dpappas/dpappas_data/models_out/test_ablation_"$1"_batch1/v3 test_emit_bioasq.json"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/media/dpappas/dpappas_data/models_out/test_ablation_"$1"_batch1/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/media/dpappas/dpappas_data/models_out/test_ablation_"$1"_batch2/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/media/dpappas/dpappas_data/models_out/test_ablation_"$1"_batch3/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_4/BioASQ-task7bPhaseB-testset4" \
"/media/dpappas/dpappas_data/models_out/test_ablation_"$1"_batch4/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_5/BioASQ-task7bPhaseB-testset5" \
"/media/dpappas/dpappas_data/models_out/test_ablation_"$1"_batch5/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"

