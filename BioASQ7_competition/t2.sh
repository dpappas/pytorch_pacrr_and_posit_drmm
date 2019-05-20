
python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/pdrmm.json" \
"MAP documents"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/bert.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/JBERT.json" \
"MAP documents"

########################################################################################

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/pdrmm_pdrmm.json" \
"MAP snippets"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/bert_pdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/JBERT.json" \
"MAP snippets"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/JBERT_F." \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/JBERT.json" \
"MAP snippets"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/pdrmm_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/pdrmm_pdrmm.json" \
"MAP snippets"

python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/bert_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/bert_pdrmm.json" \
"MAP snippets"
