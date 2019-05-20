
python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/pdrmm.json" \
"MAP documents"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/bert.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/JBERT.json" \
"MAP documents"

########################################################################################

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_1/pdrmm_pdrmm.json" \
"MAP snippets"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_1/bert_pdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/JBERT.json" \
"MAP snippets"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/JBERT_F." \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_1/JBERT.json" \
"MAP snippets"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_1/pdrmm_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_1/pdrmm_pdrmm.json" \
"MAP snippets"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_1/bert_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_1/bert_pdrmm.json" \
"MAP snippets"
