
python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/pdrmm.json" \
"MAP documents" "sig_jpdrmm_pdrmm_docs_3.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/bert.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/JBERT.json" \
"MAP documents" "sig_bert_jbert_docs_3.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_3/pdrmm_pdrmm.json" \
"MAP snippets" "sig_jpdrmm_pdrmmpdrmm_snips_3.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_3/bert_pdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/JBERT.json" \
"MAP snippets" "sig_bertpdrmm_JBERT_snips_3.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/JBERT_F." \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_3/JBERT.json" \
"MAP snippets" "sig_JBERTF_JBERT_snips_3.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_3/pdrmm_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_3/pdrmm_pdrmm.json" \
"MAP snippets" "sig_pdrmmbcnn_pdrmmpdrmm_snips_3.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_3/BioASQ-task7bPhaseB-testset3" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_3/bert_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_3/bert_pdrmm.json" \
"MAP snippets" "sig_bertbcnn_bertpdrmm_snips_3.txt"
