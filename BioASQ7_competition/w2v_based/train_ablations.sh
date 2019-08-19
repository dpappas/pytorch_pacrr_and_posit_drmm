#!/usr/bin/env bash

##############################################

CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 1 1
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 1 0 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 1 0
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 1 0 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 1 0 0 1
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 1 1 1
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 0 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 0 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 0 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 1 0 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 0 1 1 1 1 0
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 1 0 0 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 0 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 0 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 0 1 1 1
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 0 1 1 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 0 1 1 1 1
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 0 1 1 1 0 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 0 1 1 0 1 1 &
CUDA_VISIBLE_DEVICES=1 python3.6 ablation.py 1 1 0 1 0 1 1

##############################################

CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0111111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0111111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1011111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1101111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1110111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111011
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111101
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111110
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111100 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111100 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111100 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111100 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111100
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1110110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1110110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1110110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1110110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1110110
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1101110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1101110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1101110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1101110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1101110
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1011110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1011110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1011110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1011110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1011110
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0111110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0111110
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111001 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111001 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111001 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111001 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111001
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1110101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1110101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1110101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1110101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1110101
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1101101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1101101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1101101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1101101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1101101
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1011101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1011101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1011101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1011101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1011101
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0111101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0111101
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0011111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0011111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0001111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0001111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0001111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0001111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0001111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0010111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0010111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0010111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0010111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0010111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0011011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0011011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0011011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0011011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0011011
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0011101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0011101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0011101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0011101 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0011101
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0011110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0011110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0011110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0011110 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0011110
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1110011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1110011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1110011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1110011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1110011
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1100111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1100111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1100111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1100111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1100111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1010111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1010111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1010111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1010111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1010111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0110111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0110111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1101011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1101011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1101011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1101011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1101011
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1011011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1011011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1011011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1011011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1011011
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1001111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1001111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1001111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1001111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1001111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0101111 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0101111
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 0111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 0111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 0111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 0111011 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 0111011
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 1 1111010 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 2 1111010 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 3 1111010 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 4 1111010 &
CUDA_VISIBLE_DEVICES=1 python3.6 extract_ablation.py 5 1111010


# ls -d /media/dpappas/dpappas_data/models_out/ablation_* | wc -l
# ls -d /media/dpappas/dpappas_data/models_out/test_ablation_* | wc -l



# grep "v3 dev MAP documents:" -A 4 /media/dpappas/dpappas_data/models_out/sec_jpdrmm_2L_0p01_run_0/model.log
#
#java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
#"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
#"/home/dpappas/test_sec_jpdrmm_batch1/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"
#
#java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
#"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
#"/home/dpappas/test_sec_jpdrmm_batch2/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"
#
#java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
#"/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
#"/home/dpappas/test_sec_jpdrmm_batch3/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"
#
#java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
#"/home/dpappas/bioasq_all/bioasq7/data/test_batch_4/BioASQ-task7bPhaseB-testset4" \
#"/home/dpappas/test_sec_jpdrmm_batch4/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"
#
#java -Xmx10G -cp /home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar evaluation.EvaluatorTask1b -phaseA -e 5 \
#"/home/dpappas/bioasq_all/bioasq7/data/test_batch_5/BioASQ-task7bPhaseB-testset5" \
#"/home/dpappas/test_sec_jpdrmm_batch5/v3 test_emit_bioasq.json" | grep "^MAP documents:\|^MAP snippets:"


