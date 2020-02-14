
RUN=$1

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_0_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_1_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_2_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_3_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_4_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_5_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_6_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_7_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_8_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"

# ----------------------------------------------------------------------------------------------------------------

DIRI=ablation_${RUN}_0p01_9_bioasq_jpdrmm_2L_0p01_run_0

CUDA_VISIBLE_DEVICES=0 python eval_test.py 12345 /home/DISK_1/dpappas/${DIRI}/best_dev_checkpoint.pth.tar /home/DISK_1/dpappas/${DIRI}/test/

java -Xmx10G -cp "/home/DISK_1/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b  -phaseA -e 5 \
"/home/DISK_1/dpappas/bioasq_all/bioasq7/data/test_batch_12345/BioASQ-task7bPhaseB-testset12345" \
"/home/DISK_1/dpappas/${DIRI}/test/v3 test_emit_bioasq.json" | \
grep -E '^MAP snippets:|^MAP documents:' > "/home/DISK_1/dpappas/${DIRI}/test/results.txt"
