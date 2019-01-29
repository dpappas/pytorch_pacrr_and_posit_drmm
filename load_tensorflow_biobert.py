
import os
import tensorflow as tf
from pprint import pprint
from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining



bert_config_file    = '/home/dpappas/Downloads/F_BERT/Biobert/pubmed_pmc_470k/bert_config.json'
tf_checkpoint_path  = '/home/dpappas/Downloads/F_BERT/Biobert/pubmed_pmc_470k/biobert_model.ckpt'

config_path = os.path.abspath(bert_config_file)
tf_path     = os.path.abspath(tf_checkpoint_path)
print("Converting TensorFlow checkpoint from {} with config at {}".format(tf_path, config_path))
# Load weights from TF model
init_vars = tf.train.list_variables(tf_path)
names = []
arrays = []
for name, shape in init_vars:
    print("Loading TF weight {} with shape {}".format(name, shape))
    array = tf.train.load_variable(tf_path, name)
    names.append(name)
    arrays.append(array)

# Initialise PyTorch model
config = BertConfig.from_json_file(bert_config_file)
print("Building PyTorch model from configuration: {}".format(str(config)))
model = BertForPreTraining(config)

pprint(dir(model))



