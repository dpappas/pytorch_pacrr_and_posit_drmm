
import os
from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

if (not os.path.exists('/home/dpappas/bioasq_all/F_BERT/Biobert/pubmed_pmc_470k/pytorch_model.bin')):
    convert_tf_checkpoint_to_pytorch(
        '/home/dpappas/bioasq_all/F_BERT/Biobert/pubmed_pmc_470k/biobert_model.ckpt',
        '/home/dpappas/bioasq_all/F_BERT/Biobert/pubmed_pmc_470k/bert_config.json',
        '/home/dpappas/bioasq_all/F_BERT/Biobert/pubmed_pmc_470k/pytorch_model.bin'
    )


