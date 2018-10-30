#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import  os
import  json
import  time
import  random
import  logging
import  subprocess
import  numpy as np
import  torch
import  torch.nn as nn
import  torch.optim as optim
import  torch.nn.functional as F
from    pprint import pprint
import  torch.autograd as autograd
from    tqdm import tqdm
from    difflib import SequenceMatcher
from    data_handler import *

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

def dummy_test():
    doc1_embeds         = np.random.rand(40, 200)
    doc2_embeds         = np.random.rand(30, 200)
    question_embeds     = np.random.rand(20, 200)
    q_idfs              = np.random.rand(20, 1)
    gaf                 = np.random.rand(4)
    baf                 = np.random.rand(4)
    for epoch in range(200):
        optimizer.zero_grad()
        cost_, doc1_emit_, doc2_emit_ = model(
            doc1_embeds     = doc1_embeds,
            doc2_embeds     = doc2_embeds,
            question_embeds = question_embeds,
            q_idfs          = q_idfs,
            gaf             = gaf,
            baf             = baf
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

def get_map_res(fgold, femit):
    trec_eval_res   = subprocess.Popen(['python', eval_path, fgold, femit], stdout=subprocess.PIPE, shell=False)
    (out, err)      = trec_eval_res.communicate()
    lines           = out.decode("utf-8").split('\n')
    map_res         = [l for l in lines if (l.startswith('map '))][0].split('\t')
    map_res         = float(map_res[-1])
    return map_res

def back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc):
    batch_cost = sum(batch_costs) / float(len(batch_costs))
    batch_cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    batch_aver_cost = batch_cost.cpu().item()
    epoch_aver_cost = sum(epoch_costs) / float(len(epoch_costs))
    batch_aver_acc  = sum(batch_acc) / float(len(batch_acc))
    epoch_aver_acc  = sum(epoch_acc) / float(len(epoch_acc))
    return batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc

def get_bioasq_res(prefix, data_gold, data_emitted, data_for_revision):
    '''
    java -Xmx10G -cp /home/dpappas/for_ryan/bioasq6_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar
    evaluation.EvaluatorTask1b -phaseA -e 5
    /home/dpappas/for_ryan/bioasq6_submit_files/test_batch_1/BioASQ-task6bPhaseB-testset1
    ./drmm-experimental_submit.json
    '''
    jar_path = retrieval_jar_path
    #
    fgold   = '{}_data_for_revision.json'.format(prefix)
    fgold   = os.path.join(odir, fgold)
    fgold   = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_for_revision, indent=4, sort_keys=True))
        f.close()
    #
    for tt in data_gold['questions']:
        if ('exact_answer' in tt):
            del (tt['exact_answer'])
        if ('ideal_answer' in tt):
            del (tt['ideal_answer'])
        if ('type' in tt):
            del (tt['type'])
    fgold    = '{}_gold_bioasq.json'.format(prefix)
    fgold   = os.path.join(odir, fgold)
    fgold   = os.path.abspath(fgold)
    with open(fgold, 'w') as f:
        f.write(json.dumps(data_gold, indent=4, sort_keys=True))
        f.close()
    #
    femit    = '{}_emit_bioasq.json'.format(prefix)
    femit   = os.path.join(odir, femit)
    femit   = os.path.abspath(femit)
    with open(femit, 'w') as f:
        f.write(json.dumps(data_emitted, indent=4, sort_keys=True))
        f.close()
    #
    bioasq_eval_res = subprocess.Popen(
        [
            'java', '-Xmx10G', '-cp', jar_path, 'evaluation.EvaluatorTask1b',
            '-phaseA', '-e', '5', fgold, femit
        ],
        stdout=subprocess.PIPE, shell=False
    )
    (out, err)  = bioasq_eval_res.communicate()
    lines       = out.decode("utf-8").split('\n')
    ret = {}
    for line in lines:
        if(':' in line):
            k       = line.split(':')[0].strip()
            v       = line.split(':')[1].strip()
            ret[k]  = float(v)
    return ret

def similar(upstream_seq, downstream_seq):
    upstream_seq    = upstream_seq.encode('ascii','ignore')
    downstream_seq  = downstream_seq.encode('ascii','ignore')
    s               = SequenceMatcher(None, upstream_seq, downstream_seq)
    match           = s.find_longest_match(0, len(upstream_seq), 0, len(downstream_seq))
    upstream_start  = match[0]
    upstream_end    = match[0]+match[2]
    longest_match   = upstream_seq[upstream_start:upstream_end]
    to_match        = upstream_seq if(len(downstream_seq)>len(upstream_seq)) else downstream_seq
    r1              = SequenceMatcher(None, to_match, longest_match).ratio()
    return r1

def get_pseudo_retrieved(dato):
    some_ids = [item['document'].split('/')[-1].strip() for item in bioasq6_data[dato['query_id']]['snippets']]
    pseudo_retrieved            = [
        {
            'bm25_score'        : 7.76,
            'doc_id'            : id,
            'is_relevant'       : True,
            'norm_bm25_score'   : 3.85
        }
        for id in set(some_ids)
    ]
    return pseudo_retrieved

def get_snippets_loss(good_sent_tags, gs_emits_, bs_emits_):
    wright = torch.cat([gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 1)])
    wrong  = [gs_emits_[i] for i in range(len(good_sent_tags)) if (good_sent_tags[i] == 0)]
    wrong  = torch.cat(wrong + [bs_emits_.squeeze(-1)])
    losses = [ model.my_hinge_loss(w.unsqueeze(0).expand_as(wrong), wrong) for w in wright]
    return sum(losses) / float(len(losses))

def get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_):
    bs_emits_       = bs_emits_.squeeze(-1)
    gs_emits_       = gs_emits_.squeeze(-1)
    good_sent_tags  = torch.FloatTensor(good_sent_tags)
    #
    sn_d1_l         = F.binary_cross_entropy(gs_emits_, good_sent_tags, size_average=False, reduce=True)
    sn_d2_l         = F.binary_cross_entropy(bs_emits_, torch.zeros_like(bs_emits_), size_average=False, reduce=True)
    return sn_d1_l, sn_d2_l

def init_the_logger(hdlr):
    if not os.path.exists(odir):
        os.makedirs(odir)
    od          = odir.split('/')[-1] # 'sent_posit_drmm_MarginRankingLoss_0p001'
    logger      = logging.getLogger(od)
    if(hdlr is not None):
        logger.removeHandler(hdlr)
    hdlr        = logging.FileHandler(odir+'model.log')
    formatter   = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr

def train_one(epoch, bioasq6_data, two_losses=True, use_sent_tokenizer=False):
    model.train()
    batch_costs, batch_acc, epoch_costs, epoch_acc = [], [], [], []
    batch_counter = 0
    train_instances = train_data_step1(train_data)
    #
    epoch_aver_cost, epoch_aver_acc = 0., 0.
    random.shuffle(train_instances)
    #
    start_time      = time.time()
    for (
        good_sents_embeds, bad_sents_embeds, quest_embeds, q_idfs, good_sents_escores, bad_sents_escores, good_doc_af,
        bad_doc_af, good_sent_tags, bad_sent_tags, good_mesh_embeds, bad_mesh_embeds, good_mesh_escores, bad_mesh_escores
    ) in train_data_step2(train_instances, train_docs, wv, bioasq6_data, idf, max_idf, use_sent_tokenizer=use_sent_tokenizer):
        cost_, doc1_emit_, doc2_emit_, gs_emits_, bs_emits_ = model(
            doc1_sents_embeds   = good_sents_embeds,
            doc2_sents_embeds   = bad_sents_embeds,
            question_embeds     = quest_embeds,
            q_idfs              = q_idfs,
            sents_gaf           = good_sents_escores,
            sents_baf           = bad_sents_escores,
            doc_gaf             = good_doc_af,
            doc_baf             = bad_doc_af,
            good_meshes_embeds  = good_mesh_embeds,
            bad_meshes_embeds   = bad_mesh_embeds,
            mesh_gaf            = good_mesh_escores,
            mesh_baf            = bad_mesh_escores
        )
        #
        good_sent_tags, bad_sent_tags       = good_sent_tags, bad_sent_tags
        if(two_losses):
            sn_d1_l, sn_d2_l                = get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_)
            snip_loss                       = sn_d1_l + sn_d2_l
            l                               = 0.5
            cost_                           = ((1 - l) * snip_loss) + (l * cost_)
        #
        batch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_costs.append(cost_.cpu().item())
        batch_costs.append(cost_)
        if (len(batch_costs) == b_size):
            batch_counter += 1
            batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc)
            elapsed_time    = time.time() - start_time
            start_time      = time.time()
            print('{} {} {} {} {} {}'.format(batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc, elapsed_time))
            logger.info('{} {} {} {} {} {}'.format( batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc, elapsed_time))
            batch_costs, batch_acc = [], []
    if (len(batch_costs) > 0):
        batch_counter += 1
        batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs, batch_acc, epoch_acc)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('{} {} {} {} {} {}'.format(batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc, elapsed_time))
        logger.info('{} {} {} {} {} {}'.format(batch_counter, batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc, elapsed_time))
    print('Epoch:{} aver_epoch_cost: {} aver_epoch_acc: {}'.format(epoch, epoch_aver_cost, epoch_aver_acc))
    logger.info('Epoch:{} aver_epoch_cost: {} aver_epoch_acc: {}'.format(epoch, epoch_aver_cost, epoch_aver_acc))

def do_for_one_retrieved(doc_emit_, gs_emits_, held_out_sents, retr, doc_res, gold_snips):
    emition                 = doc_emit_.cpu().item()
    emitss                  = gs_emits_.tolist()
    mmax                    = max(emitss)
    all_emits, extracted_from_one = [], []
    for ind in range(len(emitss)):
        t = (snip_is_relevant(held_out_sents[ind], gold_snips), emitss[ind], "http://www.ncbi.nlm.nih.gov/pubmed/{}".format(retr['doc_id']), held_out_sents[ind])
        all_emits.append(t)
        if(emitss[ind] == mmax):
            extracted_from_one.append(t)
    doc_res[retr['doc_id']] = float(emition)
    all_emits               = sorted(all_emits, key=lambda x: x[1], reverse=True)
    return doc_res, extracted_from_one, all_emits

def do_for_some_retrieved(docs, dato, retr_docs, data_for_revision, ret_data, all_bioasq_subm_data, all_bioasq_subm_data_known, use_sent_tokenizer):
    emitions = {
        'body': dato['query_text'],
        'id': dato['query_id'],
        'documents': []
    }
    #
    quest_text                  = dato['query_text']
    quest_tokens, quest_embeds  = get_embeds(tokenize(quest_text), wv)
    q_idfs                      = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
    gold_snips                  = get_gold_snips(dato['query_id'], bioasq6_data)
    #
    doc_res, extracted_snippets, extracted_snippets_known_rel_num = {}, [], []
    for retr in retr_docs:
        datum = prep_data(quest_text, docs[retr['doc_id']], retr['norm_bm25_score'], wv, gold_snips, idf, max_idf, use_sent_tokenizer=False)
        doc_emit_, gs_emits_    = model.emit_one(
            doc1_sents_embeds   = datum['good_sents_embeds'],
            question_embeds     = quest_embeds,
            q_idfs              = q_idfs,
            sents_gaf           = datum['good_sents_escores'],
            doc_gaf             = datum['good_doc_af'],
            good_meshes_embeds  = datum['good_mesh_embeds'],
            mesh_gaf            = datum['good_mesh_escores']
        )
        doc_res, extracted_from_one, all_emits = do_for_one_retrieved(doc_emit_, gs_emits_, datum['held_out_sents'], retr, doc_res, gold_snips)
        #
        extracted_snippets.extend(extracted_from_one)
        #
        total_relevant = sum([1 for em in all_emits if(em[0]==True)])
        if (total_relevant > 0):
            extracted_snippets_known_rel_num.extend(all_emits[:total_relevant])
        if (dato['query_id'] not in data_for_revision):
            data_for_revision[dato['query_id']] = {
                'query_text': dato['query_text'],
                'snippets'  : {retr['doc_id']: all_emits}
            }
        else:
            data_for_revision[dato['query_id']]['snippets'][retr['doc_id']] = all_emits
    #
    doc_res                                 = sorted(doc_res.items(), key=lambda x: x[1], reverse=True)
    doc_res                                 = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm[0]) for pm in doc_res]
    emitions['documents']                   = doc_res[:100]
    ret_data['questions'].append(emitions)
    #
    if(use_sent_tokenizer == True):
        extracted_snippets                  = [tt for tt in extracted_snippets if (tt[2] in doc_res[:10])]
        extracted_snippets                  = sorted(extracted_snippets, key=lambda x: x[1], reverse=True)
        extracted_snippets_known_rel_num    = [tt for tt in extracted_snippets_known_rel_num if (tt[2] in doc_res[:10])]
        extracted_snippets_known_rel_num    = sorted(extracted_snippets_known_rel_num, key=lambda x: x[1], reverse=True)
    else:
        extracted_snippets                  = []
        extracted_snippets_known_rel_num    = []
    #
    snips_res                               = prep_extracted_snippets(extracted_snippets, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    all_bioasq_subm_data['questions'].append(snips_res)
    #
    snips_res_known_rel_num                 = prep_extracted_snippets(extracted_snippets_known_rel_num, docs, dato['query_id'], doc_res[:10], dato['query_text'])
    all_bioasq_subm_data_known['questions'].append(snips_res_known_rel_num)
    return data_for_revision, ret_data, all_bioasq_subm_data, all_bioasq_subm_data_known

def get_one_map(prefix, data, docs, use_sent_tokenizer=False):
    model.eval()
    #
    ret_data                    = {'questions': []}
    all_bioasq_subm_data        = {"questions": []}
    all_bioasq_subm_data_known  = {"questions": []}
    all_bioasq_gold_data        = {'questions': []}
    data_for_revision           = {}
    #
    for dato in tqdm(data['queries']):
        all_bioasq_gold_data['questions'].append(bioasq6_data[dato['query_id']])
        #
        data_for_revision, ret_data, all_bioasq_subm_data, all_bioasq_subm_data_known = do_for_some_retrieved(
            docs, dato, dato['retrieved_documents'], data_for_revision,
            ret_data, all_bioasq_subm_data, all_bioasq_subm_data_known, use_sent_tokenizer
        )
    #
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data_known, data_for_revision)
    pprint(bioasq_snip_res)
    logger.info('{} known MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    logger.info('{} known F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    logger.info('{} known MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    logger.info('{} known GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #
    bioasq_snip_res = get_bioasq_res(prefix, all_bioasq_gold_data, all_bioasq_subm_data, data_for_revision)
    pprint(bioasq_snip_res)
    logger.info('{} MAP documents: {}'.format(prefix, bioasq_snip_res['MAP documents']))
    logger.info('{} F1 snippets: {}'.format(prefix, bioasq_snip_res['F1 snippets']))
    logger.info('{} MAP snippets: {}'.format(prefix, bioasq_snip_res['MAP snippets']))
    logger.info('{} GMAP snippets: {}'.format(prefix, bioasq_snip_res['GMAP snippets']))
    #
    if (prefix == 'dev'):
        with open(odir + 'elk_relevant_abs_posit_drmm_lists_dev.json', 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(dataloc+'bioasq.dev.json', odir + 'elk_relevant_abs_posit_drmm_lists_dev.json')
    else:
        with open(odir + 'elk_relevant_abs_posit_drmm_lists_test.json', 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(dataloc+'bioasq.test.json', odir + 'elk_relevant_abs_posit_drmm_lists_test.json')
    return res_map

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim= 30, k_for_maxpool= 5, context_method = 'CNN', sentence_out_method = 'MLP', mesh_style = 'SENT'):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k                                      = k_for_maxpool
        #
        self.embedding_dim                          = embedding_dim
        self.mesh_style                             = mesh_style
        self.context_method                         = context_method
        self.sentence_out_method                    = sentence_out_method
        # to create q weights
        self.init_context_module()
        self.init_question_weight_module()
        self.init_mlps_for_pooled_attention()
        self.init_sent_output_layer()
        self.init_doc_out_layer()
        # doc loss func
        self.margin_loss                            = nn.MarginRankingLoss(margin=1.0)
    def init_mesh_module(self):
        self.mesh_h0    = autograd.Variable(torch.randn(1, 1, self.embedding_dim))
        self.mesh_gru   = nn.GRU(self.embedding_dim, self.embedding_dim)
    def init_context_module(self):
        if(self.context_method == 'CNN'):
            self.trigram_conv_1             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
            self.trigram_conv_activation_1  = torch.nn.LeakyReLU(negative_slope=0.1)
            self.trigram_conv_2             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
            self.trigram_conv_activation_2  = torch.nn.LeakyReLU(negative_slope=0.1)
        else:
            self.context_h0     = autograd.Variable(torch.randn(2, 1, self.embedding_dim))
            self.context_gru    = nn.GRU(
                input_size      = self.embedding_dim,
                hidden_size     = self.embedding_dim,
                bidirectional   = True
            )
            self.context_gru_activation = torch.nn.LeakyReLU(negative_slope=0.1)
    def init_question_weight_module(self):
        self.q_weights_mlp      = nn.Linear(self.embedding_dim+1, 1, bias=True)
    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1      = nn.Linear(3 * 3, 8, bias=True)
        self.my_relu1           = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2      = nn.Linear(8, 1, bias=True)
    def init_sent_output_layer(self):
        if(self.sentence_out_method == 'MLP'):
            self.sent_out_layer = nn.Linear(4, 1, bias=False)
        else:
            self.sent_res_h0    = autograd.Variable(torch.randn(2, 1, 5))
            self.sent_res_bigru = nn.GRU(input_size=4, hidden_size=5, bidirectional=True, batch_first=False)
            self.sent_res_mlp   = nn.Linear(10, 1, bias=False)
    def init_doc_out_layer(self):
        if(self.mesh_style=='BIGRU'):
            self.init_mesh_module()
            self.final_layer = nn.Linear(5 + 30, 1, bias=True)
        elif(self.mesh_style=='SENT'):
            self.final_layer = nn.Linear(1 + 4 + 1, 1, bias=True)
        else:
            self.final_layer = nn.Linear(5, 1, bias=True)
    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta      = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos
    def apply_context_gru(self, the_input, h0):
        output, hn      = self.context_gru(the_input.unsqueeze(1), h0)
        output          = self.context_gru_activation(output)
        out_forward     = output[:, 0, :self.embedding_dim]
        out_backward    = output[:, 0, self.embedding_dim:]
        output          = out_forward + out_backward
        res             = output + the_input
        return res, hn
    def apply_context_convolution(self, the_input, the_filters, activation):
        conv_res        = the_filters(the_input.transpose(0,1).unsqueeze(0))
        if(activation is not None):
            conv_res    = activation(conv_res)
        pad             = the_filters.padding[0]
        ind_from        = int(np.floor(pad/2.0))
        ind_to          = ind_from + the_input.size(0)
        conv_res        = conv_res[:, :, ind_from:ind_to]
        conv_res        = conv_res.transpose(1, 2)
        conv_res        = conv_res + the_input
        return conv_res.squeeze(0)
    def my_cosine_sim(self, A, B):
        A           = A.unsqueeze(0)
        B           = B.unsqueeze(0)
        A_mag       = torch.norm(A, 2, dim=2)
        B_mag       = torch.norm(B, 2, dim=2)
        num         = torch.bmm(A, B.transpose(-1,-2))
        den         = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1,-2))
        dist_mat    = num / den
        return dist_mat
    def pooling_method(self, sim_matrix):
        sorted_res              = torch.sort(sim_matrix, -1)[0]                             # sort the input minimum to maximum
        k_max_pooled            = sorted_res[:,-self.k:]                                    # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)                        # average these k values
        the_maximum             = k_max_pooled[:, -1]                                       # select the maximum value of each instance
        the_average_over_all    = sorted_res.sum(-1)/float(sim_matrix.size(1))              # add average of all elements as long sentences might have more matches
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled, the_average_over_all], dim=-1)  # concatenate maximum value and average of k-max values
        return the_concatenation     # return the concatenation
    def get_output(self, input_list, weights):
        temp    = torch.cat(input_list, -1)
        lo      = self.linear_per_q1(temp)
        lo      = self.my_relu1(lo)
        lo      = self.linear_per_q2(lo)
        lo      = lo.squeeze(-1)
        lo      = lo * weights
        sr      = lo.sum(-1) / lo.size(-1)
        return sr
    def apply_sent_res_bigru(self, the_input):
        output, hn      = self.sent_res_bigru(the_input.unsqueeze(1), self.sent_res_h0)
        output          = self.sent_res_mlp(output)
        return output.squeeze(-1).squeeze(-1)
    def do_for_one_doc_cnn(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights):
        res = []
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            conv_res            = self.apply_context_convolution(sent_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
            conv_res            = self.apply_context_convolution(conv_res,      self.trigram_conv_2, self.trigram_conv_activation_2)
            #
            sim_insens          = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_oh              = (sim_insens > (1 - (1e-3))).float()
            sim_sens            = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled  = self.pooling_method(sim_insens)
            sensitive_pooled    = self.pooling_method(sim_sens)
            oh_pooled           = self.pooling_method(sim_oh)
            #
            sent_emit           = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats      = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        if(self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        ret = self.get_max(res).unsqueeze(0)
        res = torch.sigmoid(res)
        return ret, res
    def do_for_one_doc_bigru(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights):
        res = []
        hn  = self.context_h0
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            conv_res, hn        = self.apply_context_gru(sent_embeds, hn)
            #
            sim_insens          = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_oh              = (sim_insens > (1 - (1e-3))).float()
            sim_sens            = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled  = self.pooling_method(sim_insens)
            sensitive_pooled    = self.pooling_method(sim_sens)
            oh_pooled           = self.pooling_method(sim_oh)
            #
            sent_emit           = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats      = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        if(self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        ret = self.get_max(res).unsqueeze(0)
        res = torch.sigmoid(res)
        return ret, res
    def get_max_and_average_of_k_max(self, res, k):
        sorted_res              = torch.sort(res)[0]
        k_max_pooled            = sorted_res[-k:]
        average_k_max_pooled    = k_max_pooled.sum()/float(k)
        the_maximum             = k_max_pooled[-1]
        # print(the_maximum)
        # print(the_maximum.size())
        # print(average_k_max_pooled)
        # print(average_k_max_pooled.size())
        the_concatenation       = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation
    def get_max(self, res):
        return torch.max(res)
    def get_kmax(self, res):
        res     = torch.sort(res,0)[0]
        res     = res[-self.k2:].squeeze(-1)
        if(res.size()[0] < self.k2):
            res         = torch.cat([res, torch.zeros(self.k2 - res.size()[0])], -1)
        return res
    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res
    def get_maxmin_max(self, res):
        res = self.min_max_norm(res)
        res = torch.max(res)
        return res
    def apply_mesh_gru(self, mesh_embeds):
        mesh_embeds     = autograd.Variable(torch.FloatTensor(mesh_embeds), requires_grad=False)
        output, hn      = self.mesh_gru(mesh_embeds.unsqueeze(1), self.mesh_h0)
        return output[-1,0,:]
    def get_mesh_rep(self, meshes_embeds, q_context):
        meshes_embeds   = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds   = torch.stack(meshes_embeds)
        sim_matrix      = self.my_cosine_sim(meshes_embeds, q_context).squeeze(0)
        max_sim         = torch.sort(sim_matrix, -1)[0][:, -1]
        output          = torch.mm(max_sim.unsqueeze(0), meshes_embeds)[0]
        return output
    def emit_one(self, doc1_sents_embeds, question_embeds, q_idfs, sents_gaf, doc_gaf, good_meshes_embeds, mesh_gaf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        #
        if(self.context_method=='CNN'):
            q_context       = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
            q_context       = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        else:
            q_context, _    = self.apply_context_gru(question_embeds, self.context_h0)
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        if(self.context_method=='CNN'):
            good_out, gs_emits  = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights)
        else:
            good_out, gs_emits = self.do_for_one_doc_bigru(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights)
        #
        if(self.mesh_style=='BIGRU'):
            good_meshes_out = self.get_mesh_rep(good_meshes_embeds, q_context)
            good_out_pp = torch.cat([good_out, doc_gaf, good_meshes_out], -1)
        elif (self.mesh_style == 'SENT'):
            if (self.context_method == 'CNN'):
                good_mesh_out, gs_mesh_emits = self.do_for_one_doc_cnn(good_meshes_embeds, mesh_gaf, question_embeds, q_context, q_weights)
            else:
                good_mesh_out, gs_mesh_emits = self.do_for_one_doc_bigru(good_meshes_embeds, mesh_gaf, question_embeds, q_context, q_weights)
            good_out_pp = torch.cat([good_out, doc_gaf, good_mesh_out], -1)
        else:
            good_out_pp = torch.cat([good_out, doc_gaf], -1)
        #
        final_good_output   = self.final_layer(good_out_pp)
        return final_good_output, gs_emits
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, question_embeds, q_idfs, sents_gaf, sents_baf, doc_gaf, doc_baf, good_meshes_embeds, bad_meshes_embeds, mesh_gaf, mesh_baf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        doc_baf             = autograd.Variable(torch.FloatTensor(doc_baf),             requires_grad=False)
        #
        if(self.context_method=='CNN'):
            q_context       = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
            q_context       = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        else:
            q_context, _    = self.apply_context_gru(question_embeds, self.context_h0)
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        if(self.context_method=='CNN'):
            good_out, gs_emits  = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights)
            bad_out, bs_emits   = self.do_for_one_doc_cnn(doc2_sents_embeds, sents_baf, question_embeds, q_context, q_weights)
        else:
            good_out, gs_emits  = self.do_for_one_doc_bigru(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights)
            bad_out, bs_emits   = self.do_for_one_doc_bigru(doc2_sents_embeds, sents_baf, question_embeds, q_context, q_weights)
        #
        if(self.mesh_style=='BIGRU'):
            good_meshes_out     = self.get_mesh_rep(good_meshes_embeds, q_context)
            bad_meshes_out      = self.get_mesh_rep(bad_meshes_embeds, q_context)
            good_out_pp         = torch.cat([good_out, doc_gaf, good_meshes_out], -1)
            bad_out_pp          = torch.cat([bad_out, doc_baf, bad_meshes_out], -1)
        elif(self.mesh_style=='SENT'):
            if(self.context_method=='CNN'):
                good_mesh_out, gs_mesh_emits = self.do_for_one_doc_cnn(
                    good_meshes_embeds, mesh_gaf, question_embeds, q_context, q_weights
                )
                bad_mesh_out, bs_mesh_emits = self.do_for_one_doc_cnn(
                    bad_meshes_embeds, mesh_baf, question_embeds, q_context, q_weights
                )
            else:
                good_mesh_out, gs_mesh_emits = self.do_for_one_doc_bigru(
                    good_meshes_embeds, mesh_gaf, question_embeds, q_context, q_weights
                )
                bad_mesh_out, bs_mesh_emits  = self.do_for_one_doc_bigru(
                    bad_meshes_embeds, mesh_baf, question_embeds, q_context, q_weights
                )
            good_out_pp = torch.cat([good_out, doc_gaf, good_mesh_out], -1)
            bad_out_pp  = torch.cat([bad_out, doc_baf, bad_mesh_out], -1)
        else:
            good_out_pp         = torch.cat([good_out, doc_gaf], -1)
            bad_out_pp          = torch.cat([bad_out, doc_baf], -1)
        #
        final_good_output   = self.final_layer(good_out_pp)
        final_bad_output    = self.final_layer(bad_out_pp)
        #
        loss1               = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output, gs_emits, bs_emits

# # laptop
# w2v_bin_path        = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/dpappas/for_ryan/fordp/idf.pkl'
# dataloc             = '/home/dpappas/for_ryan/'
# eval_path           = '/home/dpappas/for_ryan/eval/run_eval.py'
# retrieval_jar_path  = '/home/dpappas/NetBeansProjects/my_bioasq_eval_2/dist/my_bioasq_eval_2.jar'

# # cslab241
# w2v_bin_path        = '/home/dpappas/for_ryan/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/dpappas/for_ryan/idf.pkl'
# dataloc             = '/home/DATA/Biomedical/document_ranking/bioasq_data/'
# eval_path           = '/home/DATA/Biomedical/document_ranking/eval/run_eval.py'
# retrieval_jar_path  = '/home/dpappas/bioasq_eval/dist/my_bioasq_eval_2.jar'

# atlas , cslab243, gpu_server_1, gpu_server_2
w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
dataloc             = '/home/dpappas/bioasq_all/bioasq_data/'
eval_path           = '/home/dpappas/bioasq_all/eval/run_eval.py'
retrieval_jar_path  = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'

# w2v_bin_path        = '/home/cave/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
# idf_pickle_path     = '/home/cave/dpappas/bioasq_all/idf.pkl'
# dataloc             = '/home/cave/dpappas/bioasq_all/bioasq_data/'
# eval_path           = '/home/cave/dpappas/bioasq_all/eval/run_eval.py'
# retrieval_jar_path  = '/home/cave/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'

k_for_maxpool   = 5
k_sent_maxpool  = 2
embedding_dim   = 30 #200
lr              = 0.01
b_size          = 32
max_epoch       = 10

# model_name, context_method, sentence_method, mesh_method, two_losses_or_not
models = [
# ['Model_01', ]
# ['Model_02', ],
# ['Model_03', ],
# ['Model_04', ],
# ['Model_05', ],
# ['Model_06', ],
# ['Model_07', ],
# ['Model_08', ],
# ['Model_09', 'CNN',     'MLP',   None,      False],
# ['Model_10', 'CNN',     'MLP',   'BIGRU',   False],
# ['Model_11', 'CNN',     'BIGRU', None,      False],
# ['Model_12', 'CNN',     'BIGRU', 'BIGRU',   False],
# ['Model_13', 'BIGRU',   'MLP',   None,      False],
# ['Model_14', 'BIGRU',   'MLP',   'BIGRU',   False],
# ['Model_15', 'BIGRU',   'BIGRU', None,      False],
# ['Model_16', 'BIGRU',   'BIGRU', 'BIGRU',   False],
# ['Model_17', 'CNN',     'MLP',   None,      True],
# ['Model_18', 'CNN',     'MLP',   'BIGRU',   True],
# ['Model_19', 'CNN',     'BIGRU', None,      True],
# ['Model_20', 'CNN',     'BIGRU', 'BIGRU',   True],
# ['Model_21', 'BIGRU',   'MLP',   None,      True],
# ['Model_22', 'BIGRU',   'MLP',   'BIGRU',   True],
# ['Model_23', 'BIGRU',   'BIGRU', None,      True],
# ['Model_24', 'BIGRU',   'BIGRU', 'BIGRU',   True],
# #
# ['Model_25', 'CNN',     'MLP',   'SENT',    False],
# ['Model_26', 'CNN',     'BIGRU', 'SENT',    False],
# ['Model_27', 'BIGRU',   'MLP',   'SENT',    False],
# ['Model_28', 'BIGRU',   'BIGRU', 'SENT',    False],
# ['Model_29', 'CNN',     'MLP',   'SENT',    True],
# ['Model_30', 'CNN',     'BIGRU', 'SENT',    True],
# ['Model_31', 'BIGRU',   'MLP',   'SENT',    True],
# ['Model_32', 'BIGRU',   'BIGRU', 'SENT',    True],
#
['Model_33', 'CNN',     'MLP',      None,      False,   False],
['Model_34', 'CNN',     'MLP',      'SENT',    False,   False],
['Model_35', 'CNN',     'BIGRU',    None,      False,   False],
['Model_36', 'CNN',     'BIGRU',    'SENT',    False,   False],
['Model_37', 'BIGRU',   'MLP',      None,      False,   False],
['Model_38', 'BIGRU',   'MLP',      'SENT',    False,   False],
['Model_39', 'BIGRU',   'BIGRU',    None,      False,   False],
['Model_40', 'BIGRU',   'BIGRU',    'SENT',    False,   False],

['Model_41', 'CNN',     'MLP',      None,      False,   True],
['Model_42', 'CNN',     'MLP',      'SENT',    False,   True],
['Model_43', 'CNN',     'BIGRU',    None,      False,   True],
['Model_44', 'CNN',     'BIGRU',    'SENT',    False,   True],
['Model_45', 'BIGRU',   'MLP',      None,      False,   True],
['Model_46', 'BIGRU',   'MLP',      'SENT',    False,   True],
['Model_47', 'BIGRU',   'BIGRU',    None,      False,   True],
['Model_48', 'BIGRU',   'BIGRU',    'SENT',    False,   True],

['Model_49', 'CNN',     'MLP',      None,      True,    True],
['Model_50', 'CNN',     'MLP',      'SENT',    True,    True],
['Model_51', 'CNN',     'BIGRU',    None,      True,    True],
['Model_52', 'CNN',     'BIGRU',    'SENT',    True,    True],
['Model_53', 'BIGRU',   'MLP',      None,      True,    True],
['Model_54', 'BIGRU',   'MLP',      'SENT',    True,    True],
['Model_55', 'BIGRU',   'BIGRU',    None,      True,    True],
['Model_56', 'BIGRU',   'BIGRU',    'SENT',    True,    True],
]
models = dict([(item[0], item[1:]) for item in models])

which_model = 'Model_34'

hdlr = None
for run in range(5):
    #
    my_seed = random.randint(1, 2000000)
    random.seed(my_seed)
    torch.manual_seed(my_seed)
    #
    odir            = '/home/dpappas/{}_run_{}/'.format(which_model, run)
    #
    logger, hdlr    = init_the_logger(hdlr)
    print('random seed: {}'.format(my_seed))
    logger.info('random seed: {}'.format(my_seed))
    #
    (
        test_data, test_docs, dev_data, dev_docs, train_data,
        train_docs, idf, max_idf, wv, bioasq6_data
    ) = load_all_data(dataloc=dataloc, w2v_bin_path=w2v_bin_path, idf_pickle_path=idf_pickle_path)
    #
    print('Compiling model...')
    logger.info('Compiling model...')
    model       = Sent_Posit_Drmm_Modeler(
        embedding_dim       = embedding_dim,
        k_for_maxpool       = k_for_maxpool,
        context_method      = models[which_model][0],
        sentence_out_method = models[which_model][1],
        mesh_style          = models[which_model][2]
    )
    params      = model.parameters()
    print_params(model)
    optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #
    best_dev_map, test_map = None, None
    for epoch in range(max_epoch):
        train_one(epoch+1, bioasq6_data, two_losses=models[which_model][3], use_sent_tokenizer=models[which_model][4])
        epoch_dev_map       = get_one_map('dev', dev_data, dev_docs)
        if(best_dev_map is None or epoch_dev_map>=best_dev_map):
            best_dev_map    = epoch_dev_map
            test_map        = get_one_map('test', test_data, test_docs)
            save_checkpoint(epoch, model, best_dev_map, optimizer, filename=odir+'best_checkpoint.pth.tar')
        print('epoch:{} epoch_dev_map:{} best_dev_map:{} test_map:{}'.format(epoch + 1, epoch_dev_map, best_dev_map, test_map))
        logger.info('epoch:{} epoch_dev_map:{} best_dev_map:{} test_map:{}'.format(epoch + 1, epoch_dev_map, best_dev_map, test_map))

