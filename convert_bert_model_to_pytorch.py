
import os
from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

# in_dir          = '/media/dpappas/dpappas_data/biobert/pubmed_pmc_470k/'
in_dir          = '/media/dpappas/dpappas_data/biobert/biobert_pubmed/'
out_bin_path    = os.path.join(in_dir, 'pytorch_model.bin')

if (not os.path.exists(out_bin_path)):
    convert_tf_checkpoint_to_pytorch(os.path.join(in_dir, 'biobert_model.ckpt'), os.path.join(in_dir, 'bert_config.json'), out_bin_path)


