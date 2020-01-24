
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from importlib import reload

from deeprank.dataset import DataLoader, PairGenerator, ListGenerator
from deeprank import utils


seed = 1234
torch.manual_seed(seed)

loader = DataLoader('./config/letor07_mp_fold1.model')

import json
letor_config = json.loads(open('./config/letor07_mp_fold1.model').read())
#device = torch.device("cuda")
#device = torch.device("cpu")
select_device = torch.device("cpu")
rank_device = torch.device("cuda")

Letor07Path = letor_config['data_dir']

letor_config['fill_word'] = loader._PAD_
letor_config['embedding'] = loader.embedding
letor_config['feat_size'] = loader.feat_size
letor_config['vocab_size'] = loader.embedding.shape[0]
letor_config['embed_dim'] = loader.embedding.shape[1]
letor_config['pad_value'] = loader._PAD_

pair_gen = PairGenerator(rel_file=Letor07Path + '/relation.train.fold%d.txt'%(letor_config['fold']), config=letor_config)

from deeprank import select_module
from deeprank import rank_module

letor_config['max_match'] = 20
letor_config['win_size'] = 5
select_net = select_module.QueryCentricNet(config=letor_config, out_device=rank_device)
select_net = select_net.to(select_device)
select_net.train()

letor_config["dim_q"] = 1
letor_config["dim_d"] = 1
letor_config["dim_weight"] = 1
letor_config["c_reduce"] = [1, 1]
letor_config["k_reduce"] = [1, 50]
letor_config["s_reduce"] = 1
letor_config["p_reduce"] = [0, 0]

letor_config["c_en_conv_out"] = 4
letor_config["k_en_conv"] = 3
letor_config["s_en_conv"] = 1
letor_config["p_en_conv"] = 1

letor_config["en_pool_out"] = [1, 1]
letor_config["en_leaky"] = 0.2

letor_config["dim_gru_hidden"] = 3

letor_config['lr'] = 0.005
letor_config['finetune_embed'] = False

rank_net = rank_module.DeepRankNet(config=letor_config)
rank_net = rank_net.to(rank_device)
rank_net.embedding.weight.data.copy_(torch.from_numpy(loader.embedding))
rank_net.qw_embedding.weight.data.copy_(torch.from_numpy(loader.idf_embedding))
rank_net.train()
rank_optimizer = optim.Adam(rank_net.parameters(), lr=letor_config['lr'])

def to_device(*variables, device):
    return (torch.from_numpy(variable).to(device) for variable in variables)

def show_text(x):
    print(' '.join([loader.word_dict[w.item()] for w in x]))

X1, X1_len, X1_id, X2, X2_len, X2_id, Y, F = pair_gen.get_batch(data1=loader.query_data, data2=loader.doc_data)
X1, X1_len, X2, X2_len, Y, F = to_device(X1, X1_len, X2, X2_len, Y, F, device=rank_device)

show_text(X2[0])

X1, X2_new, X1_len, X2_len_new, X2_pos = select_net(X1, X2, X1_len, X2_len, X1_id, X2_id)

show_text(X1[0])
for i in range(5):
    print(i, end=' ')
    show_text(X2_new[0][i])

print(X2_pos[20].shape)
print(len(X2_pos))
print(len(X2))
print(X2_pos[0])
print(X2_pos[1])

import time
rank_loss_list = []
start_t = time.time()
for i in range(1000):
    # One Step Forward
    X1, X1_len, X1_id, X2, X2_len, X2_id, Y, F = pair_gen.get_batch(data1=loader.query_data, data2=loader.doc_data)
    X1, X1_len, X2, X2_len, Y, F = to_device(X1, X1_len, X2, X2_len, Y, F, device=select_device)
    X1, X2, X1_len, X2_len, X2_pos = select_net(X1, X2, X1_len, X2_len, X1_id, X2_id)
    X2, X2_len = utils.data_adaptor(X2, X2_len, select_net, rank_net, letor_config)
    output = rank_net(X1, X2, X1_len, X2_len, X2_pos)
    # Update Rank Net
    rank_loss = rank_net.pair_loss(output, Y)
    print('rank loss:', rank_loss.item())
    rank_loss_list.append(rank_loss.item())
    rank_optimizer.zero_grad()
    rank_loss.backward()
    rank_optimizer.step()

end_t = time.time()
print('Time Cost: %s s' % (end_t - start_t))

torch.save(select_net, "qcentric.model")
torch.save(rank_net, "deeprank.model")





