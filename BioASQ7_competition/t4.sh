
python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset123" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b123_joined/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b123_joined/pdrmm.json" \
"MAP documents" "sig_jpdrmm_pdrmm_docs_123.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset123" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b123_joined/bert.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b123_joined/JBERT.json" \
"MAP documents" "sig_bert_jbert_docs_123.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset123" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b123_joined/jpdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b123_joined/pdrmm_pdrmm.json" \
"MAP snippets" "sig_jpdrmm_pdrmmpdrmm_snips_123.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset123" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b123_joined/bert_pdrmm.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b123_joined/JBERT.json" \
"MAP snippets" "sig_bertpdrmm_JBERT_snips_123.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset123" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b123_joined/JBERT_F.json" \
"/home/dpappas/bioasq_all/bioasq7/document_results/b123_joined/JBERT.json" \
"MAP snippets" "sig_JBERTF_JBERT_snips_123.txt" &
python3.6 sig.py  \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset123" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b123_joined/pdrmm_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b123_joined/pdrmm_pdrmm.json" \
"MAP snippets" "sig_pdrmmbcnn_pdrmmpdrmm_snips_123.txt" &
python3.6 sig.py \
"/home/dpappas/bioasq_all/bioasq7/data/BioASQ-task7bPhaseB-testset123" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b123_joined/bert_bcnn.json" \
"/home/dpappas/bioasq_all/bioasq7/snippet_results/b123_joined/bert_pdrmm.json" \
"MAP snippets" "sig_bertbcnn_bertpdrmm_snips_123.txt"
