
import torch
from pytorch_transformers import *
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import  torch.nn.functional as F

def encode_sents_faster_but_padding(sents):
    ###############################################################
    tokenized_sents = [bert_tokenizer.encode(sent) for sent in sents]
    max_len         = max(len(sent) for sent in tokenized_sents)
    pad_id          = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.pad_token)
    input_ids       = torch.tensor([sent_ids + ([pad_id] * (max_len - len(sent_ids))) for sent_ids in tokenized_sents])
    ###############################################################
    with torch.no_grad():
        last_hidden_state, pooler_output, hidden_states, attentions = bert_model(input_ids)
    ###############################################################
    # print(last_hidden_state.size())
    # print(pooler_output.size())
    # print(len(hidden_states))
    # print([t.size() for t in hidden_states])
    # print(len(attentions))
    # print([t.size() for t in attentions])
    ###############################################################
    return last_hidden_state, pooler_output, hidden_states, attentions

def encode_sents(sents):
    last_hidden_state, pooler_output = [], []
    for sent in  sents:
        tokenized_sent = bert_tokenizer.encode(sent,add_special_tokens=True)
        input_ids       = torch.tensor([tokenized_sent])
        with torch.no_grad():
            _last_hidden_state, _pooler_output = bert_model(input_ids)
            last_hidden_state.append(_last_hidden_state)
            pooler_output.append(_pooler_output)
    return last_hidden_state, torch.cat(pooler_output, dim=0)

class MLP(nn.Module):
    def __init__(self, input_dim=None, sizes=None, activation_functions=None, initializer_range=0.02):
        super(MLP, self).__init__()
        ################################
        sizes               = [input_dim]+ sizes
        self.activations    = activation_functions
        self.linears        = []
        self.initializer_range = initializer_range
        for i in range(len(sizes)-1):
            self.linears.append(nn.Linear(sizes[i], sizes[i+1]))
        ################################
        self.apply(self.init_weights)
        for lin in self.linears:
            lin.apply(self.init_weights)
        ################################
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def forward(self, features):
        ret = features
        for layer, activation in zip(self.linears, self.activations):
            ret = layer(ret)
            if(activation is not None):
                ret = activation(ret)
        return ret

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class JBert(nn.Module):
    def __init__(self, embedding_dim=768, initializer_range=0.02):
        super(JBert, self).__init__()
        self.sent_add_feats = 10
        self.initializer_range = initializer_range
        leaky_relu_lambda   = lambda t: F.leaky_relu(t, negative_slope=0.1)
        self.snip_MLP       = MLP(input_dim=embedding_dim,          sizes=[100, 1], activation_functions=[leaky_relu_lambda, torch.sigmoid])
        self.doc_MLP        = MLP(input_dim=self.sent_add_feats+1,  sizes=[8, 1],   activation_functions=[leaky_relu_lambda, torch.sigmoid])
        self.apply(self.init_weights)
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, doc1_sents_af, doc2_sents_af, doc1_af, doc2_af):
        doc1_sents_int_score    = self.snip_MLP(doc1_sents_embeds)
        doc2_sents_int_score    = self.snip_MLP(doc2_sents_embeds)
        #########################
        doc1_int_sent_scores_af = torch.cat([doc1_sents_int_score, doc1_sents_af], -1)
        doc2_int_sent_scores_af = torch.cat([doc2_sents_int_score, doc2_sents_af], -1)
        #########################
        sents1_out              = torch.sigmoid(self.snip_mlp_2(doc1_int_sent_scores_af))
        sents2_out              = torch.sigmoid(self.snip_mlp_2(doc2_int_sent_scores_af))
        #########################
        max_feats_of_sents_1    = torch.max(sents1_out, 0)[0].unsqueeze(0)
        max_feats_of_sents_1_af = torch.cat([max_feats_of_sents_1, doc1_af], -1)
        max_feats_of_sents_2    = torch.max(sents2_out, 0)[0].unsqueeze(0)
        max_feats_of_sents_2_af = torch.cat([max_feats_of_sents_2, doc2_af], -1)
        #########################
        doc1_out                = self.doc_MLP(max_feats_of_sents_1_af)
        doc2_out                = self.doc_MLP(max_feats_of_sents_2_af)
        #########################
        return doc1_out, sents1_out, doc2_out, sents2_out

def print_params(model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(model)
    print(40 * '=')
    trainable = 0
    untrainable = 0
    for parameter in list(model.parameters()):
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        if (parameter.requires_grad):
            trainable += v
        else:
            untrainable += v
    total_params = trainable + untrainable
    print(40 * '=')
    print('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    print(40 * '=')

# (model_class, tokenizer_class, pretrained_weights) = (BertModel, BertTokenizer, 'bert-base-uncased')
(model_class, tokenizer_class, pretrained_weights) = (RobertaModel, RobertaTokenizer, 'roberta-base')

bert_tokenizer  = tokenizer_class.from_pretrained(pretrained_weights)
# bert_model      = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
bert_model      = model_class.from_pretrained(pretrained_weights, output_hidden_states=False, output_attentions=False)
model           = JBert(768)

print_params(bert_model)
print_params(model)
sents           = ["Here is some text to encode", "Here is another text to see whether size matters"]
last_hidden_state, pooler_output = encode_sents(sents)

# initializer_range   = 0.02
# lr                  = 1e-3
# max_grad_norm       = 1.0
# num_total_steps     = 1000
# num_warmup_steps    = 100
# warmup_proportion   = float(num_warmup_steps) / float(num_total_steps)  # 0.1
# ### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
# optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
# scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
# ### and used like this:
# for batch in train_data:
#     loss = model(batch)
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
#     optimizer.step()
#     scheduler.step()
#     optimizer.zero_grad()

