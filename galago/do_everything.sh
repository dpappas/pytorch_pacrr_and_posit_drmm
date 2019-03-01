
python3.6 queries2galago.py \
/home/DATA/bioasq7/BioASQ-training7b/trainining7b.json \
./bioasq7b_all_galago_queries.json

/home/DATA/Biomedical/document_ranking/bioasq_data/document_retrieval/galago-3.10-bin/bin/galago \
batch-search \
--index=pubmed_only_abstract_galago_index \
--verbose=False \
--requested=2000 \
--scorer=bm25 \
--defaultTextPart=postings.krovetz \
--mode=threaded \
./bioasq7b_all_galago_queries.json \
> ./bioasq7b_bm25_retrieval.all.txt

# grep " 2000 " bioasq7b_bm25_retrieval.all.txt | wc -l # to see progress

python3.6 /home/dpappas/generate_bioasq_data.py \
trainining7b.json \
bioasq7b_bm25_retrieval.all.txt \
all \
2016


