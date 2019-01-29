
import os
from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

init_checkpoint_pt  = "/home/dpappas/Downloads/F_BERT/Biobert/pubmed_pmc_470k/"

if (not os.path.exists(os.path.join(init_checkpoint_pt, 'pytorch_model.bin'))):
    convert_tf_checkpoint_to_pytorch(
        '/home/dpappas/Downloads/F_BERT/Biobert/pubmed_pmc_470k/biobert_model.ckpt',
        '/home/dpappas/Downloads/F_BERT/Biobert/pubmed_pmc_470k/bert_config.json',
        '/home/dpappas/Downloads/F_BERT/Biobert/pubmed_pmc_470k/pytorch_model.bin'
    )
