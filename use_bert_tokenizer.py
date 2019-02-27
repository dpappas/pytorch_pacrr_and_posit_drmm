

from    pytorch_pretrained_bert.tokenization import BertTokenizer
cache_dir           = '/home/dpappas/bert_cache/'
bert_tokenizer_file = '/home/dpappas/for_ryan/F_BERT/Biobert/pubmed_pmc_470k/vocab.txt'
bert_tokenizer      = BertTokenizer.from_pretrained(
    bert_tokenizer_file,
    do_lower_case   = True,
    cache_dir       = cache_dir
)


q   = 'Are microtubules marked by glutamylation?'
s   = 'deposition of polymodifications (glutamylation and glycylation)'
print(bert_tokenizer.tokenize(q))
print(bert_tokenizer.tokenize(s))

