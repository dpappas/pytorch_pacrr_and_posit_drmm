python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset1234" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/pdrmm.json" \
"MAP documents" "sig_jpdrmm_pdrmm_docs_1234.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset1234" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/bert.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/JBERT.json" \
"MAP documents" "sig_bert_jbert_docs_1234.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset1234" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b1234_joined/pdrmm_pdrmm.json" \
"MAP snippets" "sig_jpdrmm_pdrmmpdrmm_snips_1234.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset1234" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b1234_joined/bert_pdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/JBERT.json" \
"MAP snippets" "sig_bertpdrmm_JBERT_snips_1234.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset1234" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/JBERT_F.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b1234_joined/JBERT.json" \
"MAP snippets" "sig_JBERTF_JBERT_snips_1234.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset1234" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b1234_joined/pdrmm_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b1234_joined/pdrmm_pdrmm.json" \
"MAP snippets" "sig_pdrmmbcnn_pdrmmpdrmm_snips_1234.txt" &
python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset1234" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b1234_joined/bert_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b1234_joined/bert_pdrmm.json" \
"MAP snippets" "sig_bertbcnn_bertpdrmm_snips_1234.txt"