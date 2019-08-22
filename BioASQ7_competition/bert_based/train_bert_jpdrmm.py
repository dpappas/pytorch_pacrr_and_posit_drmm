#!/usr/bin/env python
# -*- coding: utf-8 -*-

import  time
import  torch.nn.functional         as F
import  torch.nn                    as nn
import  torch.optim                 as optim
import  torch.autograd              as autograd
from    pytorch_pretrained_bert.tokenization import BertTokenizer
from    pytorch_pretrained_bert.modeling import BertForSequenceClassification
from    pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from    BioASQ7_competition.bert_based.bert_needed_functions import *

def train_data_step2(instances, docs, bioasq6_data, idf, max_idf, use_sent_tokenizer):
    for quest_text, quest_id, gid, bid, bm25s_gid, bm25s_bid in instances:
        ####
        good_snips          = get_snips(quest_id, gid, bioasq6_data)
        good_snips          = [' '.join(bioclean(sn)) for sn in good_snips]
        quest_text          = ' '.join(bioclean(quest_text.replace('\ufeff', ' ')))
        quest_tokens, qemb  = embed_the_sent(quest_text)
        q_idfs              = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
        ####
        datum               = prep_data(quest_text, docs[gid], bm25s_gid, good_snips, idf, max_idf, quest_tokens)
        good_sents_embeds   = datum['sents_embeds']
        good_sents_escores  = datum['sents_escores']
        good_doc_af         = datum['doc_af']
        good_sent_tags      = datum['sent_tags']
        good_held_out_sents = datum['held_out_sents']
        good_oh_sims        = datum['oh_sims']
        #
        datum               = prep_data(quest_text, docs[bid], bm25s_bid, [], idf, max_idf, quest_tokens)
        bad_sents_embeds    = datum['sents_embeds']
        bad_sents_escores   = datum['sents_escores']
        bad_doc_af          = datum['doc_af']
        bad_sent_tags       = [0] * len(datum['sent_tags'])
        bad_held_out_sents  = datum['held_out_sents']
        bad_oh_sims         = datum['oh_sims']
        #
        if (use_sent_tokenizer == False or sum(good_sent_tags) > 0):
            yield {
                'good_sents_embeds': good_sents_embeds,
                'good_sents_escores': good_sents_escores,
                'good_doc_af': good_doc_af,
                'good_sent_tags': good_sent_tags,
                'good_held_out_sents': good_held_out_sents,
                'good_oh_sims': good_oh_sims,
                #
                'bad_sents_embeds': bad_sents_embeds,
                'bad_sents_escores': bad_sents_escores,
                'bad_doc_af': bad_doc_af,
                'bad_sent_tags': bad_sent_tags,
                'bad_held_out_sents': bad_held_out_sents,
                'bad_oh_sims': bad_oh_sims,
                #
                'quest_embeds': qemb,
                'q_idfs': q_idfs,
            }

def train_one(epoch, bioasq6_data, two_losses, use_sent_tokenizer):
    model.train()
    bert_model.train()
    batch_costs, batch_acc, epoch_costs, epoch_acc = [], [], [], []
    batch_counter, epoch_aver_cost, epoch_aver_acc = 0, 0., 0.
    #
    train_instances = train_data_step1(train_data)
    random.shuffle(train_instances)
    #
    start_time = time.time()
    pbar = tqdm(
        iterable= train_data_step2(train_instances, train_docs, bioasq6_data, idf, max_idf, use_sent_tokenizer),
        total   = 14288, #9684, # 378,
        ascii   = True
    )
    for datum in pbar:
        cost_, doc1_emit_, doc2_emit_, gs_emits_, bs_emits_ = model(
            doc1_sents_embeds=datum['good_sents_embeds'],
            doc2_sents_embeds=datum['bad_sents_embeds'],
            doc1_oh_sim=datum['good_oh_sims'],
            doc2_oh_sim=datum['bad_oh_sims'],
            question_embeds=datum['quest_embeds'],
            q_idfs=datum['q_idfs'],
            sents_gaf=datum['good_sents_escores'],
            sents_baf=datum['bad_sents_escores'],
            doc_gaf=datum['good_doc_af'],
            doc_baf=datum['bad_doc_af']
        )
        #
        good_sent_tags, bad_sent_tags = datum['good_sent_tags'], datum['bad_sent_tags']
        if (two_losses):
            sn_d1_l, sn_d2_l = get_two_snip_losses(good_sent_tags, gs_emits_, bs_emits_)
            snip_loss = sn_d1_l + sn_d2_l
            l = 0.5
            cost_ = ((1 - l) * snip_loss) + (l * cost_)
        #
        batch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_acc.append(float(doc1_emit_ > doc2_emit_))
        epoch_costs.append(cost_.cpu().item())
        batch_costs.append(cost_)
        if (len(batch_costs) == b_size):
            batch_counter += 1
            batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs,
                                                                                         batch_acc, epoch_acc)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                     batch_aver_acc, epoch_aver_acc, elapsed_time))
            logger.info(
                '{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                   batch_aver_acc, epoch_aver_acc, elapsed_time))
            batch_costs, batch_acc = [], []
    if (len(batch_costs) > 0):
        batch_counter += 1
        batch_aver_cost, epoch_aver_cost, batch_aver_acc, epoch_aver_acc = back_prop(batch_costs, epoch_costs,
                                                                                     batch_acc, epoch_acc)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                 batch_aver_acc, epoch_aver_acc, elapsed_time))
        logger.info('{:03d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(batch_counter, batch_aver_cost, epoch_aver_cost,
                                                                       batch_aver_acc, epoch_aver_acc, elapsed_time))
    print('Epoch:{:02d} aver_epoch_cost: {:.4f} aver_epoch_acc: {:.4f}'.format(epoch, epoch_aver_cost, epoch_aver_acc))
    logger.info(
        'Epoch:{:02d} aver_epoch_cost: {:.4f} aver_epoch_acc: {:.4f}'.format(epoch, epoch_aver_cost, epoch_aver_acc))

def get_one_map(prefix, data, docs, use_sent_tokenizer):
    model.eval()
    bert_model.eval()
    #
    ret_data = {'questions': []}
    all_bioasq_subm_data_v1 = {"questions": []}
    all_bioasq_subm_data_known_v1 = {"questions": []}
    all_bioasq_subm_data_v2 = {"questions": []}
    all_bioasq_subm_data_known_v2 = {"questions": []}
    all_bioasq_subm_data_v3 = {"questions": []}
    all_bioasq_subm_data_known_v3 = {"questions": []}
    all_bioasq_gold_data = {'questions': []}
    data_for_revision = {}
    #
    for dato in tqdm(data['queries'], ascii=True):
        all_bioasq_gold_data['questions'].append(bioasq6_data[dato['query_id']])
        data_for_revision, ret_data, snips_res, snips_res_known = do_for_some_retrieved(docs, dato,
                                                                                        dato['retrieved_documents'],
                                                                                        data_for_revision, ret_data,
                                                                                        use_sent_tokenizer)
        all_bioasq_subm_data_v1['questions'].append(snips_res['v1'])
        all_bioasq_subm_data_v2['questions'].append(snips_res['v2'])
        all_bioasq_subm_data_v3['questions'].append(snips_res['v3'])
        all_bioasq_subm_data_known_v1['questions'].append(snips_res_known['v1'])
        all_bioasq_subm_data_known_v2['questions'].append(snips_res_known['v3'])
        all_bioasq_subm_data_known_v3['questions'].append(snips_res_known['v3'])
    #
    print_the_results('v1 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v1, all_bioasq_subm_data_known_v1,
                      data_for_revision)
    print_the_results('v2 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v2, all_bioasq_subm_data_known_v2,
                      data_for_revision)
    print_the_results('v3 ' + prefix, all_bioasq_gold_data, all_bioasq_subm_data_v3, all_bioasq_subm_data_known_v3,
                      data_for_revision)
    #
    if (prefix == 'dev'):
        with open(os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(
            os.path.join(odir, 'v3 dev_gold_bioasq.json'),
            os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_dev.json')
        )
    else:
        with open(os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_test.json'), 'w') as f:
            f.write(json.dumps(ret_data, indent=4, sort_keys=True))
        res_map = get_map_res(
            os.path.join(odir, 'v3 test_gold_bioasq.json'),
            os.path.join(odir, 'elk_relevant_abs_posit_drmm_lists_test.json')
        )
    return res_map

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim=30, k_for_maxpool=5, sentence_out_method='MLP', k_sent_maxpool=1):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k = k_for_maxpool
        self.k_sent_maxpool = k_sent_maxpool
        self.doc_add_feats = 11
        self.sent_add_feats = 10
        #
        self.embedding_dim = embedding_dim
        self.sentence_out_method = sentence_out_method
        # to create q weights
        self.init_context_module()
        self.init_question_weight_module()
        self.init_mlps_for_pooled_attention()
        self.init_sent_output_layer()
        self.init_doc_out_layer()
        # doc loss func
        self.margin_loss = nn.MarginRankingLoss(margin=1.0).to(device)

    def init_mesh_module(self):
        self.mesh_h0 = autograd.Variable(torch.randn(1, 1, self.embedding_dim)).to(device)
        self.mesh_gru = nn.GRU(self.embedding_dim, self.embedding_dim).to(device)

    def init_context_module(self):
        self.trigram_conv_1 = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True).to(device)
        self.trigram_conv_activation_1 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
        self.trigram_conv_2 = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True).to(device)
        self.trigram_conv_activation_2 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)

    def init_question_weight_module(self):
        self.q_weights_mlp = nn.Linear(self.embedding_dim + 1, 1, bias=True).to(device)

    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1 = nn.Linear(3 * 3, 8, bias=True).to(device)
        self.my_relu1 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
        self.linear_per_q2 = nn.Linear(8, 1, bias=True).to(device)

    def init_sent_output_layer(self):
        if (self.sentence_out_method == 'MLP'):
            self.sent_out_layer_1 = nn.Linear(self.sent_add_feats + 1, 8, bias=False).to(device)
            self.sent_out_activ_1 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
            self.sent_out_layer_2 = nn.Linear(8, 1, bias=False).to(device)
        else:
            self.sent_res_h0 = autograd.Variable(torch.randn(2, 1, 5)).to(device)
            self.sent_res_bigru = nn.GRU(input_size=self.sent_add_feats + 1, hidden_size=5, bidirectional=True,
                                         batch_first=False).to(device)
            self.sent_res_mlp = nn.Linear(10, 1, bias=False).to(device)

    def init_doc_out_layer(self):
        self.final_layer_1 = nn.Linear(self.doc_add_feats + self.k_sent_maxpool, 8, bias=True).to(device)
        self.final_activ_1 = torch.nn.LeakyReLU(negative_slope=0.1).to(device)
        self.final_layer_2 = nn.Linear(8, 1, bias=True).to(device)
        self.oo_layer = nn.Linear(2, 1, bias=True).to(device)

    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos

    def apply_context_gru(self, the_input, h0):
        output, hn = self.context_gru(the_input.unsqueeze(1), h0)
        output = self.context_gru_activation(output)
        out_forward = output[:, 0, :self.embedding_dim]
        out_backward = output[:, 0, self.embedding_dim:]
        output = out_forward + out_backward
        res = output + the_input
        return res, hn

    def apply_context_convolution(self, the_input, the_filters, activation):
        conv_res = the_filters(the_input.transpose(0, 1).unsqueeze(0))
        if (activation is not None):
            conv_res = activation(conv_res)
        pad = the_filters.padding[0]
        ind_from = int(np.floor(pad / 2.0))
        ind_to = ind_from + the_input.size(0)
        conv_res = conv_res[:, :, ind_from:ind_to]
        conv_res = conv_res.transpose(1, 2)
        conv_res = conv_res + the_input
        return conv_res.squeeze(0)

    def my_cosine_sim(self, A, B):
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        A_mag = torch.norm(A, 2, dim=2)
        B_mag = torch.norm(B, 2, dim=2)
        num = torch.bmm(A, B.transpose(-1, -2))
        den = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
        dist_mat = num / den
        return dist_mat

    def pooling_method(self, sim_matrix):
        sorted_res = torch.sort(sim_matrix, -1)[0]  # sort the input minimum to maximum
        k_max_pooled = sorted_res[:, -self.k:]  # select the last k of each instance in our data
        average_k_max_pooled = k_max_pooled.sum(-1) / float(self.k)  # average these k values
        the_maximum = k_max_pooled[:, -1]  # select the maximum value of each instance
        the_average_over_all = sorted_res.sum(-1) / float(
            sim_matrix.size(1))  # add average of all elements as long sentences might have more matches
        the_concatenation = torch.stack([the_maximum, average_k_max_pooled, the_average_over_all],
                                        dim=-1)  # concatenate maximum value and average of k-max values
        return the_concatenation  # return the concatenation

    def get_output(self, input_list, weights):
        temp = torch.cat(input_list, -1)
        lo = self.linear_per_q1(temp)
        lo = self.my_relu1(lo)
        lo = self.linear_per_q2(lo)
        lo = lo.squeeze(-1)
        lo = lo * weights
        sr = lo.sum(-1) / lo.size(-1)
        return sr

    def apply_sent_res_bigru(self, the_input):
        output, hn = self.sent_res_bigru(the_input.unsqueeze(1), self.sent_res_h0)
        output = self.sent_res_mlp(output)
        return output.squeeze(-1).squeeze(-1)

    def do_for_one_doc_cnn(self, doc_sents_embeds, oh_sims, sents_af, question_embeds, q_conv_res_trigram, q_weights,
                           k2):
        res = []
        for i in range(len(doc_sents_embeds)):
            sim_oh = autograd.Variable(torch.FloatTensor(oh_sims[i]), requires_grad=False).to(device)
            sent_embeds = doc_sents_embeds[i]
            gaf = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False).to(device)
            #
            conv_res            = self.apply_context_convolution(sent_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
            conv_res            = self.apply_context_convolution(conv_res, self.trigram_conv_2, self.trigram_conv_activation_2)
            #
            sim_insens          = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
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
        if (self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer_1(res)
            res = self.sent_out_activ_1(res)
            res = self.sent_out_layer_2(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        # ret = self.get_max(res).unsqueeze(0)
        ret = self.get_kmax(res, k2)
        return ret, res

    def do_for_one_doc_bigru(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights, k2):
        res = []
        hn = self.context_h0
        for i in range(len(doc_sents_embeds)):
            sent_embeds = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False).to(device)
            gaf = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False).to(device)
            conv_res, hn = self.apply_context_gru(sent_embeds, hn)
            #
            sim_insens = self.my_cosine_sim(question_embeds, sent_embeds).squeeze(0)
            sim_oh = (sim_insens > (1 - (1e-3))).float()
            sim_sens = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
            #
            insensitive_pooled = self.pooling_method(sim_insens)
            sensitive_pooled = self.pooling_method(sim_sens)
            oh_pooled = self.pooling_method(sim_oh)
            #
            sent_emit = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
            sent_add_feats = torch.cat([gaf, sent_emit.unsqueeze(-1)])
            res.append(sent_add_feats)
        res = torch.stack(res)
        if (self.sentence_out_method == 'MLP'):
            res = self.sent_out_layer_1(res)
            res = self.sent_out_activ_1(res)
            res = self.sent_out_layer_2(res).squeeze(-1)
        else:
            res = self.apply_sent_res_bigru(res)
        # ret = self.get_max(res).unsqueeze(0)
        ret = self.get_kmax(res, k2)
        res = torch.sigmoid(res)
        return ret, res

    def get_max(self, res):
        return torch.max(res)

    def get_kmax(self, res, k):
        res = torch.sort(res, 0)[0]
        res = res[-k:].squeeze(-1)
        if (len(res.size()) == 0):
            res = res.unsqueeze(0)
        if (res.size()[0] < k):
            to_concat = torch.zeros(k - res.size()[0]).to(device)
            res = torch.cat([res, to_concat], -1)
        return res

    def get_max_and_average_of_k_max(self, res, k):
        k_max_pooled = self.get_kmax(res, k)
        average_k_max_pooled = k_max_pooled.sum() / float(k)
        the_maximum = k_max_pooled[-1]
        the_concatenation = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation

    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res

    def get_maxmin_max(self, res):
        res = self.min_max_norm(res)
        res = torch.max(res)
        return res

    def apply_mesh_gru(self, mesh_embeds):
        mesh_embeds = autograd.Variable(torch.FloatTensor(mesh_embeds), requires_grad=False).to(device)
        output, hn = self.mesh_gru(mesh_embeds.unsqueeze(1), self.mesh_h0)
        return output[-1, 0, :]

    def get_mesh_rep(self, meshes_embeds, q_context):
        meshes_embeds = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds = torch.stack(meshes_embeds)
        sim_matrix = self.my_cosine_sim(meshes_embeds, q_context).squeeze(0)
        max_sim = torch.sort(sim_matrix, -1)[0][:, -1]
        output = torch.mm(max_sim.unsqueeze(0), meshes_embeds)[0]
        return output

    def emit_one(self, doc1_sents_embeds, doc1_oh_sim, question_embeds, q_idfs, sents_gaf, doc_gaf):
        q_idfs          = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False).to(device)
        doc_gaf         = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False).to(device)
        #
        q_context = self.apply_context_convolution(question_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context = self.apply_context_convolution(q_context, self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights = torch.cat([q_context, q_idfs], -1)
        q_weights = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits = self.do_for_one_doc_cnn(
            doc1_sents_embeds,
            doc1_oh_sim,
            sents_gaf,
            question_embeds,
            q_context,
            q_weights,
            self.k_sent_maxpool
        )
        #
        good_out_pp = torch.cat([good_out, doc_gaf], -1)
        #
        final_good_output = self.final_layer_1(good_out_pp)
        final_good_output = self.final_activ_1(final_good_output)
        final_good_output = self.final_layer_2(final_good_output)
        #
        gs_emits = gs_emits.unsqueeze(-1)
        gs_emits = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits = torch.sigmoid(gs_emits)
        #
        return final_good_output, gs_emits

    def forward(self, doc1_sents_embeds, doc2_sents_embeds, doc1_oh_sim, doc2_oh_sim,
                question_embeds, q_idfs, sents_gaf, sents_baf, doc_gaf, doc_baf):
        q_idfs = autograd.Variable(torch.FloatTensor(q_idfs), requires_grad=False).to(device)
        doc_gaf = autograd.Variable(torch.FloatTensor(doc_gaf), requires_grad=False).to(device)
        doc_baf = autograd.Variable(torch.FloatTensor(doc_baf), requires_grad=False).to(device)
        #
        q_context = self.apply_context_convolution(question_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context = self.apply_context_convolution(q_context, self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights = torch.cat([q_context, q_idfs], -1)
        q_weights = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits = self.do_for_one_doc_cnn(
            doc1_sents_embeds,
            doc1_oh_sim,
            sents_gaf,
            question_embeds,
            q_context,
            q_weights,
            self.k_sent_maxpool
        )
        bad_out, bs_emits = self.do_for_one_doc_cnn(
            doc2_sents_embeds,
            doc2_oh_sim,
            sents_baf,
            question_embeds,
            q_context,
            q_weights,
            self.k_sent_maxpool
        )
        #
        good_out_pp = torch.cat([good_out, doc_gaf], -1)
        bad_out_pp = torch.cat([bad_out, doc_baf], -1)
        #
        final_good_output = self.final_layer_1(good_out_pp)
        final_good_output = self.final_activ_1(final_good_output)
        final_good_output = self.final_layer_2(final_good_output)
        #
        gs_emits = gs_emits.unsqueeze(-1)
        gs_emits = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits = torch.sigmoid(gs_emits)
        #
        final_bad_output = self.final_layer_1(bad_out_pp)
        final_bad_output = self.final_activ_1(final_bad_output)
        final_bad_output = self.final_layer_2(final_bad_output)
        #
        bs_emits = bs_emits.unsqueeze(-1)
        bs_emits = torch.cat([bs_emits, final_good_output.unsqueeze(-1).expand_as(bs_emits)], -1)
        bs_emits = self.oo_layer(bs_emits).squeeze(-1)
        bs_emits = torch.sigmoid(bs_emits)
        #
        loss1 = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output, gs_emits, bs_emits

# atlas , cslab243
#####################
eval_path           = '/home/dpappas/bioasq_all/eval/run_eval.py'
retrieval_jar_path  = '/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar'
odd                 = '/home/dpappas/'
#####################
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
dataloc             = '/home/dpappas/bioasq_all/bioasq7_data/'
#####################
bert_all_words_path = '/home/dpappas/bioasq_all/bert_all_words.pkl'
#####################
use_cuda            = True
max_seq_length      = 50
device              = torch.device("cuda") if(use_cuda) else torch.device("cpu")
bert_model          = 'bert-base-uncased'
cache_dir           = '/home/dpappas/bert_cache/'
bert_tokenizer      = BertTokenizer.from_pretrained(bert_model, do_lower_case=True, cache_dir=cache_dir)
bert_model          = BertForSequenceClassification.from_pretrained(bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1), num_labels=2)
bert_model.to(device)
#####################
k_for_maxpool       = 5
k_sent_maxpool      = 5
embedding_dim       = 768 # 50  # 30  # 200
lr                  = 0.01
b_size              = 32
max_epoch           = 4
#####################

(dev_data, dev_docs, train_data, train_docs, idf, max_idf, bioasq6_data) = load_all_data(dataloc=dataloc, idf_pickle_path=idf_pickle_path)

hdlr = None
for run in range(0, 5):
    #
    my_seed = run
    random.seed(my_seed)
    torch.manual_seed(my_seed)
    #
    odir = 'bioasq7_bert_jpdrmm_2L_0p01_run_{}/'.format(run)
    odir = os.path.join(odd, odir)
    print(odir)
    if (not os.path.exists(odir)):
        os.makedirs(odir)
    #
    logger, hdlr = init_the_logger(hdlr)
    print('random seed: {}'.format(my_seed))
    logger.info('random seed: {}'.format(my_seed))
    #
    avgdl, mean, deviation = get_bm25_metrics(avgdl=21.2508, mean=0.5973, deviation=0.5926)
    print(avgdl, mean, deviation)
    #
    print('Compiling model...')
    logger.info('Compiling model...')
    model = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool).to(device)
    params      = list(model.parameters())
    params      += list(bert_model.parameters())
    print_params(model)
    optimizer   = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    #
    best_dev_map, test_map = None, None
    for epoch in range(max_epoch):
        train_one(epoch + 1, bioasq6_data, two_losses=True, use_sent_tokenizer=True)
        epoch_dev_map = get_one_map('dev', dev_data, dev_docs, use_sent_tokenizer=True)
        if (best_dev_map is None or epoch_dev_map >= best_dev_map):
            best_dev_map = epoch_dev_map
            save_checkpoint(epoch, model,      best_dev_map, optimizer, filename=os.path.join(odir, 'best_checkpoint.pth.tar'))
            save_checkpoint(epoch, bert_model, best_dev_map, optimizer, filename=os.path.join(odir, 'best_bert_checkpoint.pth.tar'))
        print('epoch:{:02d} epoch_dev_map:{:.4f} best_dev_map:{:.4f}'.format(epoch + 1, epoch_dev_map, best_dev_map))
        logger.info('epoch:{:02d} epoch_dev_map:{:.4f} best_dev_map:{:.4f}'.format(epoch + 1, epoch_dev_map, best_dev_map))

'''
['[CLS]', 'the', 'drugs', 'covered', 'target', 'ga', '##ba', '##a', 'za', '##le', '##pl', '##on', '-', 'cr', 'lore', '##di', '##pl', '##on', 'ev', '##t', '-', '201', 'ore', '##xin', 'fi', '##lore', '##xa', '##nt', 'min', '-', '202', 'his', '##tam', '##ine', '-', 'h', '##1', 'l', '##y', '##26', '##24', '##80', '##3', 'ser', '##oton', '##in', '5', '-', 'h', '##t', '##2', '##a', 'it', '##i', '-', '00', '##7', 'mel', '##aton', '##ins', '##ero', '##ton', '##in', '##5', '-', 'h', '##t', '##1', '##a', 'pi', '##rom', '##ela', '##tine', 'and', 'mel', '##aton', '##in', 'indication', 'expansion', '##s', 'of', 'prolonged', '-', 'release', 'mel', '##aton', '##in', 'and', 'ta', '##si', '##mel', '##te', '##on', 'for', 'pediatric', 'sleep', 'and', 'circa', '##dian', '[SEP]']
len([
'the', 'drugs', 'covered', 'target', 'gabaa', 'zaleplon', '-', 'cr', 'lorediplon', 'evt', '-', '201', 'orexin', 
'filorexant', 'min', '-', '202', 'histamine', '-', 'h1', 'ly2624803', 'serotonin', '5', '-', 'ht2a', 'iti', '-', 
'007', 'melatoninserotonin5', '-', 'ht1a', 'piromelatine', 'and', 'melatonin', 'indication', 'expansions', 
'of', 'prolonged', '-', 'release', 'melatonin', 'and', 'tasimelteon', 'for', 'pediatric', 'sleep', 'and', 
'circadian'
])

['the', 'drugs', 'covered', 'target', 'gabaa', 'zaleplon-cr', 'lorediplon', 'evt-201', 'orexin', 
'filorexant', 'min-202', 'histamine-h1', 'ly2624803', 'serotonin', '5-ht2a', 'iti-007',
'melatoninserotonin5-ht1a', 'piromelatine', 'and', 'melatonin', 'indication', 'expansions', 
'of', 'prolonged-release', 'melatonin', 'and', 'tasimelteon', 'for', 'pediatric', 'sleep', 'and', 'circadian', 
'rhythm', 'disorders', 'receptors']

python3.6
import pickle
from pprint import pprint
d = pickle.load(open('/home/dpappas/bioasq_all/bert_elmo_embeds/25423562.p','rb'))
pprint(list(d.keys()))

pprint(d['abs_bert_original_tokens'])

[
['[CLS]', 'introduction', 'ins', '##om', '##nia', 'is', 'ty', '##pi', '##fied', 'by', 'a', 'difficulty', 'in', 'sleep', 'initiation', 'maintenance', 'and', '##or', 'quality', 'non', '-', 'rest', '##ora', '##tive', 'sleep', 'resulting', 'in', 'significant', 'daytime', 'distress', '[SEP]'], 
['[CLS]', 'areas', 'covered', 'this', 'review', 'sum', '##mar', '##izes', 'the', 'available', 'efficacy', 'and', 'safety', 'data', 'for', 'drugs', 'currently', 'in', 'the', 'pipeline', 'for', 'treating', 'ins', '##om', '##nia', '[SEP]'], 
['[CLS]', 'specifically', 'the', 'authors', 'performed', 'med', '##line', 'and', 'internet', 'searches', 'using', 'the', 'key', '##words', 'phase', 'ii', 'and', 'ins', '##om', '##nia', '[SEP]'], 
['[CLS]', 'the', 'drugs', 'covered', 'target', 'ga', '##ba', '##a', 'za', '##le', '##pl', '##on', '-', 'cr', 'lore', '##di', '##pl', '##on', 'ev', '##t', '-', '201', 'ore', '##xin', 'fi', '##lore', '##xa', '##nt', 'min', '-', '202', 'his', '##tam', '##ine', '-', 'h', '##1', 'l', '##y', '##26', '##24', '##80', '##3', 'ser', '##oton', '##in', '5', '-', 'h', '##t', '##2', '##a', 'it', '##i', '-', '00', '##7', 'mel', '##aton', '##ins', '##ero', '##ton', '##in', '##5', '-', 'h', '##t', '##1', '##a', 'pi', '##rom', '##ela', '##tine', 'and', 'mel', '##aton', '##in', 'indication', 'expansion', '##s', 'of', 'prolonged', '-', 'release', 'mel', '##aton', '##in', 'and', 'ta', '##si', '##mel', '##te', '##on', 'for', 'pediatric', 'sleep', 'and', 'circa', '##dian', '[SEP]'], 
['[CLS]', 'expert', 'opinion', 'low', '-', 'priced', 'generic', 'environments', 'and', 'high', 'development', 'costs', 'limit', 'the', 'further', 'development', 'of', 'drugs', 'that', 'treat', 'ins', '##om', '##nia', '[SEP]'], 
['[CLS]', 'however', 'the', 'bid', '##ire', '##ction', '##al', 'link', 'between', 'sleep', 'and', 'certain', 'como', '##rb', '##idi', '##ties', 'may', 'encourage', 'development', 'of', 'specific', 'drugs', 'for', 'como', '##rb', '##id', 'ins', '##om', '##nia', '[SEP]'], 
['[CLS]', 'new', 'ins', '##om', '##nia', 'the', '##ra', '##pies', 'will', 'most', 'likely', 'move', 'away', 'from', 'ga', '##ba', '##ar', 'receptors', 'modulation', 'to', 'more', 'subtle', 'neurological', 'pathways', 'that', 'regulate', 'the', 'sleep', '-', 'wake', 'cycle', '[SEP]']
]

'''
