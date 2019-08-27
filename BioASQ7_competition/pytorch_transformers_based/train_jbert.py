
import torch
from pytorch_transformers import *

(model_class, tokenizer_class, pretrained_weights) = (BertModel, BertTokenizer, 'bert-base-uncased')
# (model_class, tokenizer_class, pretrained_weights) = (RobertaModel, RobertaTokenizer, 'roberta-base')

tokenizer   = tokenizer_class.from_pretrained(pretrained_weights)
model       = model_class.from_pretrained(pretrained_weights)

sents           = ["Here is some text to encode", "Here is another text to see whether size matters"]
tokenized_sents = [tokenizer.encode(sent) for sent in sents]
max_len         = max(len(sent) for sent in tokenized_sents)
pad_id          = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
input_ids       = torch.tensor(
    [
        sent_ids + ([pad_id] * (max_len - len(sent_ids)))
        for sent_ids in tokenized_sents
    ]
)

with torch.no_grad():
    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

print(last_hidden_states.size())






