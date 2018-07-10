
import os
import re
import sys
import json
import random
import subprocess
import numpy as np
import cPickle as pickle
from pprint import pprint
from nltk.tokenize import sent_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from tqdm import tqdm

my_seed = 1989
random.seed(my_seed)
torch.manual_seed(my_seed)

odir            = '/home/dpappas/simplest_posit_drmm_2/'
if not os.path.exists(odir):
    os.makedirs(odir)

od              = 'sent_posit_drmm_MarginRankingLoss'
k_for_maxpool   = 5
lr              = 0.001
bsize           = 32
reg_lambda      = 0.1

import logging
logger = logging.getLogger(od)
hdlr = logging.FileHandler(odir+'model.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

print('LOADING embedding_matrix (14GB)...')
logger.info('LOADING embedding_matrix (14GB)...')
matrix          = np.load('/home/dpappas/joint_task_list_batches/embedding_matrix.npy')
# idf_mat         = np.load('/home/dpappas/joint_task_list_batches/idf_matrix.npy')
# print(idf_mat.shape)
# matrix          = np.random.random((150, 10))
# idf_mat          = np.random.random((150))
print(matrix.shape)

def get_index(token, t2i):
    try:
        return t2i[token]
    except KeyError:
        return t2i['UNKN']

def get_sim_mat(stoks, qtoks):
    sm = np.zeros((len(stoks), len(qtoks)))
    for i in range(len(qtoks)):
        for j in range(len(stoks)):
            if(qtoks[i] == stoks[j]):
                sm[j,i] = 1.
    return sm

def get_item_inds(item, question, t2i):
    passage     = item['title'] + ' ' + item['abstractText']
    all_sims    = get_sim_mat(bioclean(passage), bioclean(question))
    sents_inds  = [get_index(token, t2i) for token in bioclean(passage)]
    quest_inds  = [get_index(token, t2i) for token in bioclean(question)]
    return sents_inds, quest_inds, all_sims

def print_params(model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(model)
    print(40 * '=')
    logger.info(40 * '=')
    logger.info(model)
    logger.info(40 * '=')
    trainable       = 0
    untrainable     = 0
    for parameter in model.parameters():
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        if(parameter.requires_grad):
            trainable   += v
        else:
            untrainable += v
    total_params = trainable + untrainable
    print(40 * '=')
    print('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    print(40 * '=')
    logger.info(40 * '=')
    logger.info('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    logger.info(40 * '=')

def data_yielder(bm25_scores, all_abs, t2i, how_many_loops):
    for quer in bm25_scores[u'queries']:
        quest       = quer['query_text']
        ret_pmids   = [t[u'doc_id'] for t in quer[u'retrieved_documents']]
        good_pmids  = [t for t in ret_pmids if t in quer[u'relevant_documents']]
        bad_pmids   = [t for t in ret_pmids if t not in quer[u'relevant_documents']]
        if(len(bad_pmids)>0):
            for i in range(how_many_loops):
                for gid in good_pmids:
                    # bid                                             = bad_pmids[i%len(bad_pmids)]
                    bid                                             = random.choice(bad_pmids)
                    good_sents_inds, good_quest_inds, good_all_sims = get_item_inds(all_abs[gid], quest, t2i)
                    bad_sents_inds, bad_quest_inds, bad_all_sims    = get_item_inds(all_abs[bid], quest, t2i)
                    yield good_sents_inds, good_all_sims, bad_sents_inds, bad_all_sims, bad_quest_inds

def dummy_test():
    quest_inds          = np.random.randint(0,100,(40))
    good_sents_inds     = np.random.randint(0,100,(36))
    good_all_sims       = np.zeros((36, 40))
    bad_sents_inds      = np.random.randint(0,100,(37))
    bad_all_sims        = np.zeros((37, 40))
    for epoch in range(200):
        optimizer.zero_grad()
        cost_, doc1_emit_, doc2_emit_, loss1_, loss2_ = model(
            doc1        = good_sents_inds,
            doc2        = bad_sents_inds,
            question    = quest_inds,
            doc1_sim    = good_all_sims,
            doc2_sim    = bad_all_sims
        )
        cost_.backward()
        optimizer.step()
        the_cost = cost_.cpu().item()
        print(the_cost, float(doc1_emit_), float(doc2_emit_))
    print(20 * '-')

def compute_the_cost(costs, back_prop=True):
    cost_ = torch.stack(costs)
    cost_ = cost_.sum() / (1.0 * cost_.size(0))
    if(back_prop):
        cost_.backward()
        optimizer.step()
        optimizer.zero_grad()
    the_cost = cost_.cpu().item()
    return the_cost

def save_checkpoint(epoch, model, max_dev_map, optimizer, filename='checkpoint.pth.tar'):
    '''
    :param state:       the stete of the pytorch mode
    :param filename:    the name of the file in which we will store the model.
    :return:            Nothing. It just saves the model.
    '''
    state = {
        'epoch':            epoch,
        'state_dict':       model.state_dict(),
        'best_valid_score': max_dev_map,
        'optimizer':        optimizer.state_dict(),
    }
    torch.save(state, filename)

def train_one(train_instances):
    costs   = []
    optimizer.zero_grad()
    instance_metr, average_total_loss, average_task_loss, average_reg_loss = 0.0, 0.0, 0.0, 0.0
    for good_sents_inds, good_all_sims, bad_sents_inds, bad_all_sims, quest_inds in train_instances:
        instance_cost, doc1_emit, doc2_emit, loss1, loss2 = model(good_sents_inds, bad_sents_inds, quest_inds, good_all_sims, bad_all_sims)
        #
        average_total_loss  += instance_cost.cpu().item()
        average_task_loss   += loss1.cpu().item()
        average_reg_loss    += loss2.cpu().item()
        #
        instance_metr       += 1
        costs.append(instance_cost)
        if(len(costs) == bsize):
            batch_loss      = compute_the_cost(costs, True)
            costs = []
            print('train epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch,instance_metr,average_total_loss/(1.*instance_metr),average_task_loss/(1.*instance_metr),average_reg_loss/(1.*instance_metr)))
            logger.info('train epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch,instance_metr,average_total_loss/(1.*instance_metr),average_task_loss/(1.*instance_metr),average_reg_loss/(1.*instance_metr)))
    if(len(costs)>0):
        batch_loss = compute_the_cost(costs, True)
        print('train epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch, instance_metr, average_total_loss/(1.*instance_metr), average_task_loss/(1.*instance_metr), average_reg_loss/(1.*instance_metr)))
        logger.info('train epoch:{}, batch:{}, average_total_loss:{}, average_task_loss:{}, average_reg_loss:{}'.format(epoch, instance_metr, average_total_loss/(1.*instance_metr), average_task_loss/(1.*instance_metr), average_reg_loss/(1.*instance_metr)))
    return average_task_loss / instance_metr

def get_one_map(prefix, bm25_scores, all_abs):
    data = {}
    data['questions'] = []
    for quer in tqdm(bm25_scores['queries']):
        dato = {'body': quer['query_text'],'id': quer['query_id'],'documents': []}
        doc_res = {}
        for retr in quer['retrieved_documents']:
            doc_id      = retr['doc_id']
            passage     = all_abs[doc_id]['title'] + ' ' + all_abs[doc_id]['abstractText']
            all_sims    = get_sim_mat(bioclean(passage), bioclean(quer['query_text']))
            sents_inds  = [get_index(token, t2i) for token in bioclean(passage)]
            quest_inds  = [get_index(token, t2i) for token in bioclean(quer['query_text'])]
            doc1_emit_  = model.emit_one(doc1=sents_inds, question=quest_inds, doc1_sim=all_sims)
            doc_res[doc_id] = float(doc1_emit_)
        doc_res = sorted(doc_res.items(), key=lambda x: x[1], reverse=True)
        doc_res = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
        dato['documents'] = doc_res
        data['questions'].append(dato)
    if(prefix=='dev'):
        with open(odir + 'elk_relevant_abs_posit_drmm_lists_dev.json', 'w') as f:
            f.write(json.dumps(data, indent=4, sort_keys=True))
        res_map = get_map_res(
            '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq.dev.json',
            odir+'elk_relevant_abs_posit_drmm_lists_dev.json'
        )
    else:
        with open(odir + 'elk_relevant_abs_posit_drmm_lists_test.json', 'w') as f:
            f.write(json.dumps(data, indent=4, sort_keys=True))
        res_map = get_map_res(
            '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq.test.json',
            odir+'elk_relevant_abs_posit_drmm_lists_test.json'
        )
    return res_map

def load_data():
    print('Loading abs texts...')
    logger.info('Loading abs texts...')
    train_all_abs   = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.train.pkl', 'rb'))
    dev_all_abs     = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.dev.pkl', 'rb'))
    test_all_abs    = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.test.pkl', 'rb'))
    print('Loading retrieved docsc...')
    logger.info('Loading retrieved docsc...')
    train_bm25_scores   = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.train.pkl', 'rb'))
    dev_bm25_scores     = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.dev.pkl', 'rb'))
    test_bm25_scores    = pickle.load(open('/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.test.pkl', 'rb'))
    print('Loading token to index files...')
    logger.info('Loading token to index files...')
    token_to_index_f = '/home/dpappas/joint_task_list_batches/t2i.p'
    t2i = pickle.load(open(token_to_index_f, 'rb'))
    print('yielding data')
    logger.info('yielding data')
    return train_all_abs, dev_all_abs, test_all_abs, train_bm25_scores, dev_bm25_scores, test_bm25_scores, t2i

def get_map_res(fgold, femit):
    trec_eval_res = subprocess.Popen(
        [
            'python',
            '/home/DATA/Biomedical/document_ranking/eval/run_eval.py',
            fgold,
            femit
        ],
        stdout=subprocess.PIPE, shell=False)
    (out, err) = trec_eval_res.communicate()
    map_res = float([ l for l in out.split('\n') if(l.startswith('map '))][0].split('\t')[-1])
    return map_res

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, pretrained_embeds, k_for_maxpool):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k                                      = k_for_maxpool         # k is for the average k pooling
        #
        self.vocab_size                             = pretrained_embeds.shape[0]
        self.embedding_dim                          = pretrained_embeds.shape[1]
        self.word_embeddings                        = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeds))
        self.word_embeddings.weight.requires_grad   = False
        #
        self.trigram_conv                           = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2)
        self.trigram_conv_activation                = torch.nn.Sigmoid()
        #
        self.linear_per_q1                          = nn.Linear(6, 8, bias=True)
        self.linear_per_q2                          = nn.Linear(8, 1, bias=True)
        self.my_relu1                               = torch.nn.LeakyReLU()
        self.my_relu2                               = torch.nn.LeakyReLU()
        self.my_drop1                               = nn.Dropout(p=0.2)
        self.margin_loss                            = nn.MarginRankingLoss(margin=0.9)
    def apply_convolution(self, the_input, the_filters, activation):
        conv_res    = the_filters(the_input.transpose(0,1).unsqueeze(0))
        conv_res    = activation(conv_res)
        pad         = the_filters.padding[0]
        ind_from    = int(np.floor(pad/2.0))
        ind_to      = ind_from + the_input.size(0)
        conv_res    = conv_res[:, :, ind_from:ind_to]
        conv_res    = conv_res.transpose(1, 2)
        conv_res    = conv_res + the_input
        return conv_res.squeeze(0)
    def my_cosine_sim(self,A,B):
        A           = A.unsqueeze(0)
        B           = B.unsqueeze(0)
        A_mag       = torch.norm(A, 2, dim=2)
        B_mag       = torch.norm(B, 2, dim=2)
        num         = torch.bmm(A, B.transpose(-1,-2))
        den         = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1,-2))
        dist_mat    = num / den
        return dist_mat
    def pooling_method(self, sim_matrix):
        sorted_res              = torch.sort(sim_matrix, -1)[0]             # sort the input minimum to maximum
        k_max_pooled            = sorted_res[:,-self.k:]                    # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)        # average these k values
        the_maximum             = k_max_pooled[:, -1]                       # select the maximum value of each instance
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled], dim=-1) # concatenate maximum value and average of k-max values
        return the_concatenation     # return the concatenation
    def apply_masks_on_similarity(self, document, question, similarity):
        qq = (question > 1).float()
        ss              = (document > 1).float()
        sim_mask1       = qq.unsqueeze(-1).expand_as(similarity)
        sim_mask2       = ss.unsqueeze(0).expand_as(similarity)
        similarity      *= sim_mask1
        similarity      *= sim_mask2
        return similarity
    def get_output(self, input_list):
        temp    = torch.cat(input_list, -1)
        lo      = self.linear_per_q1(temp)
        lo      = self.my_relu1(lo)
        lo      = self.my_drop1(lo)
        lo      = self.linear_per_q2(lo)
        # lo      = F.sigmoid(lo)
        lo      = self.my_relu2(lo)
        lo      = lo.squeeze(-1)
        sr      = lo.sum(-1) / lo.size(-1)
        return sr
    def get_reg_loss(self):
        l2_reg = None
        for W in self.parameters():
            if(W.requires_grad):
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)
        return l2_reg
    def emit_one(self, doc1, question, doc1_sim):
        question                        = autograd.Variable(torch.LongTensor(question), requires_grad=False)
        doc1                            = autograd.Variable(torch.LongTensor(doc1),     requires_grad=False)
        sim_oh_d1                       = autograd.Variable(torch.FloatTensor(doc1_sim).transpose(0,1), requires_grad=False)
        question_embeds                 = self.word_embeddings(question)
        doc1_embeds                     = self.word_embeddings(doc1)
        sim_insensitive_d1              = self.my_cosine_sim(question_embeds, doc1_embeds).squeeze(0)
        q_conv_res_trigram              = self.apply_convolution(question_embeds, self.trigram_conv, self.trigram_conv_activation)
        d1_conv_trigram                 = self.apply_convolution(doc1_embeds,     self.trigram_conv, self.trigram_conv_activation)
        sim_sensitive_d1_trigram        = self.my_cosine_sim(q_conv_res_trigram, d1_conv_trigram).squeeze(0)
        sim_insensitive_pooled_d1       = self.pooling_method(sim_insensitive_d1)
        sim_sensitive_pooled_d1_trigram = self.pooling_method(sim_sensitive_d1_trigram)
        sim_oh_pooled_d1                = self.pooling_method(sim_oh_d1)
        doc1_emit                       = self.get_output([sim_oh_pooled_d1, sim_insensitive_pooled_d1, sim_sensitive_pooled_d1_trigram])
        return doc1_emit
    def forward(self, doc1, doc2, question, doc1_sim, doc2_sim):
        question                        = autograd.Variable(torch.LongTensor(question), requires_grad=False)
        doc1                            = autograd.Variable(torch.LongTensor(doc1),     requires_grad=False)
        doc2                            = autograd.Variable(torch.LongTensor(doc2),     requires_grad=False)
        #
        sim_oh_d1                       = autograd.Variable(torch.FloatTensor(doc1_sim).transpose(0,1), requires_grad=False)
        sim_oh_d2                       = autograd.Variable(torch.FloatTensor(doc2_sim).transpose(0,1), requires_grad=False)
        #
        question_embeds                 = self.word_embeddings(question)
        doc1_embeds                     = self.word_embeddings(doc1)
        doc2_embeds                     = self.word_embeddings(doc2)
        #
        sim_insensitive_d1              = self.my_cosine_sim(question_embeds, doc1_embeds).squeeze(0)
        sim_insensitive_d2              = self.my_cosine_sim(question_embeds, doc2_embeds).squeeze(0)
        #
        q_conv_res_trigram              = self.apply_convolution(question_embeds, self.trigram_conv, self.trigram_conv_activation)
        d1_conv_trigram                 = self.apply_convolution(doc1_embeds,     self.trigram_conv, self.trigram_conv_activation)
        d2_conv_trigram                 = self.apply_convolution(doc2_embeds,     self.trigram_conv, self.trigram_conv_activation)
        #
        sim_sensitive_d1_trigram        = self.my_cosine_sim(q_conv_res_trigram, d1_conv_trigram).squeeze(0)
        sim_sensitive_d2_trigram        = self.my_cosine_sim(q_conv_res_trigram, d2_conv_trigram).squeeze(0)
        #
        sim_insensitive_pooled_d1       = self.pooling_method(sim_insensitive_d1)
        sim_sensitive_pooled_d1_trigram = self.pooling_method(sim_sensitive_d1_trigram)
        sim_oh_pooled_d1                = self.pooling_method(sim_oh_d1)
        #
        sim_insensitive_pooled_d2       = self.pooling_method(sim_insensitive_d2)
        sim_sensitive_pooled_d2_trigram = self.pooling_method(sim_sensitive_d2_trigram)
        sim_oh_pooled_d2                = self.pooling_method(sim_oh_d2)
        #
        doc1_emit                       = self.get_output([sim_oh_pooled_d1, sim_insensitive_pooled_d1, sim_sensitive_pooled_d1_trigram])
        doc2_emit                       = self.get_output([sim_oh_pooled_d2, sim_insensitive_pooled_d2, sim_sensitive_pooled_d2_trigram])
        #
        loss1                           = self.margin_loss(doc1_emit.unsqueeze(0), doc2_emit.unsqueeze(0), torch.ones(1))
        # loss2                           = self.get_reg_loss() * reg_lambda
        loss2                           = loss1 * 0.
        loss                            = loss1 #+ loss2 + loss3
        return loss, doc1_emit, doc2_emit, loss1, loss2

print('Compiling model...')
logger.info('Compiling model...')
model  = Sent_Posit_Drmm_Modeler(pretrained_embeds=matrix, k_for_maxpool=k_for_maxpool)
params = list(set(model.parameters()) - set([model.word_embeddings.weight]))
print_params(model)
del(matrix)
optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# dummy_test()
# exit()

train_all_abs, dev_all_abs, test_all_abs, train_bm25_scores, dev_bm25_scores, test_bm25_scores, t2i = load_data()

t1 = dict(item for item in train_all_abs.items() if(int(item[1]['publicationDate'].split('-')[0]))<2017)
print(len(t1), len(train_all_abs))
t1 = dict(item for item in dev_all_abs.items() if(int(item[1]['publicationDate'].split('-')[0]))<2017)
print(len(t1), len(dev_all_abs))
t1 = dict(item for item in test_all_abs.items() if(int(item[1]['publicationDate'].split('-')[0]))<2017)
print(len(t1), len(test_all_abs))

for item in dev_bm25_scores['queries']:
    rd = [t['doc_id'] for t in item['retrieved_documents']]
    rd = [t for t in rd if(int(dev_all_abs[t]['publicationDate'].split('-')[0])>2016)]
    if(len(rd)>0):
        print rd



max_dev_map     = 0.0
max_epochs      = 30
loopes          = [1,0,0]
for epoch in range(max_epochs):
    train_instances         = data_yielder(train_bm25_scores, train_all_abs, t2i, loopes[0])
    train_average_loss      = train_one(train_instances)
    dev_map                 = get_one_map('dev', dev_bm25_scores, dev_all_abs)
    if(max_dev_map < dev_map):
        max_dev_map         = dev_map
        min_loss_epoch      = epoch+1
        test_map            = get_one_map('test', test_bm25_scores, test_all_abs)
        save_checkpoint(epoch, model, max_dev_map, optimizer, filename=odir+'best_checkpoint.pth.tar')
    print("epoch:{}, train_average_loss:{}, dev_map:{}, test_map:{}".format(epoch+1, train_average_loss, dev_map, test_map))
    print(20 * '-')
    logger.info("epoch:{}, train_average_loss:{}, dev_map:{}, test_map:{}".format(epoch+1, train_average_loss, dev_map, test_map))
    logger.info(20 * '-')



