
import torch
from pytorch_transformers import *

(model_class, tokenizer_class, pretrained_weights) = (BertModel, BertTokenizer, 'bert-base-uncased')
(model_class, tokenizer_class, pretrained_weights) = (RobertaModel, RobertaTokenizer, 'roberta-base')

tokenizer   = tokenizer_class.from_pretrained(pretrained_weights)
model       = model_class.from_pretrained(pretrained_weights)

# Encode text
input_ids = torch.tensor([tokenizer.encode("Here is some text to encode")])
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

print(last_hidden_states.size())






