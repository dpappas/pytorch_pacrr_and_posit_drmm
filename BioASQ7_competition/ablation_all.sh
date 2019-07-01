#!/usr/bin/env bash






CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 1 0








java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/media/dpappas/dpappas_data/models_out/test_ablation_1111100_batch1/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/media/dpappas/dpappas_data/models_out/test_ablation_1111100_batch2/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/media/dpappas/dpappas_data/models_out/test_ablation_1111100_batch3/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_4/BioASQ-task7bPhaseB-testset4" \
"/media/dpappas/dpappas_data/models_out/test_ablation_1111100_batch4/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"

java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_5/BioASQ-task7bPhaseB-testset5" \
"/media/dpappas/dpappas_data/models_out/test_ablation_1111100_batch5/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"






