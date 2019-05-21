
python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/pdrmm.json" \
"MAP documents" "sig_jpdrmm_pdrmm_docs_2.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/bert.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/JBERT.json" \
"MAP documents" "sig_bert_jbert_docs_2.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/pdrmm_pdrmm.json" \
"MAP snippets" "sig_jpdrmm_pdrmmpdrmm_snips_2.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/bert_pdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/JBERT.json" \
"MAP snippets" "sig_bertpdrmm_JBERT_snips_2.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/JBERT_F.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_2/JBERT.json" \
"MAP snippets" "sig_JBERTF_JBERT_snips_2.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/pdrmm_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/pdrmm_pdrmm.json" \
"MAP snippets" "sig_pdrmmbcnn_pdrmmpdrmm_snips_2.txt"

python3.6 sig.py "/home/dpappas/bioasq_all/bioasq7/data/test_batch_2/BioASQ-task7bPhaseB-testset2" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/bert_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_2/bert_pdrmm.json" \
"MAP snippets" "sig_bertbcnn_bertpdrmm_snips_2.txt"
