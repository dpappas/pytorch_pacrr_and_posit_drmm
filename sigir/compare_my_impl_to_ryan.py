import heapq
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import pickle, random, sys, re, os
import numpy as np

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def tokenize(x):
  return bioclean(x)

def GetWords(data, doc_text, words):
  for i in range(len(data['queries'])):
    qwds = tokenize(data['queries'][i]['query_text'])
    for w in qwds:
      words[w] = 1
    for j in range(len(data['queries'][i]['retrieved_documents'])):
      doc_id = data['queries'][i]['retrieved_documents'][j]['doc_id']
      dtext = (
              doc_text[doc_id]['title'] + ' <title> ' + doc_text[doc_id]['abstractText'] +
              ' '.join(
                  [
                      ' '.join(mm) for mm in
                      get_the_mesh(doc_text[doc_id])
                  ]
              )
      )
      dwds = tokenize(dtext)
      for w in dwds:
        words[w] = 1

def load_model_from_checkpoint(resume_from):
    global start_epoch, optimizer
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, embedding_dim= 30, k_for_maxpool= 5):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k                                      = k_for_maxpool
        #
        self.doc_add_feats                          = 11
        self.embedding_dim                          = embedding_dim
        # to create q weights
        self.init_context_module()
        self.init_question_weight_module()
        self.init_mlps_for_pooled_attention()
        self.init_doc_out_layer()
        # doc loss func
        self.margin_loss                            = nn.MarginRankingLoss(margin=1.0)
    def init_context_module(self):
        self.trigram_conv_1             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_1  = torch.nn.LeakyReLU(negative_slope=0.1)
        self.trigram_conv_2             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        self.trigram_conv_activation_2  = torch.nn.LeakyReLU(negative_slope=0.1)
    def init_question_weight_module(self):
        self.q_weights_mlp      = nn.Linear(self.embedding_dim+1, 1, bias=True)
    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1      = nn.Linear(3 * 3, 8, bias=True)
        self.my_relu1           = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2      = nn.Linear(8, 1, bias=True)
    def init_doc_out_layer(self):
        self.final_layer = nn.Linear(self.doc_add_feats+1, 1, bias=True)
    def init_sent_output_layer(self):
        if(self.context_method == 'MLP'):
            self.sent_out_layer = nn.Linear(4, 1, bias=False)
        else:
            self.sent_res_h0    = autograd.Variable(torch.randn(2, 1, 5))
            self.sent_res_bigru = nn.GRU(input_size=4, hidden_size=5, bidirectional=True, batch_first=False)
            self.sent_res_mlp   = nn.Linear(10, 1, bias=False)
    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta      = negatives - positives
        loss_q_pos = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos
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
        res = self.sent_out_layer(res).squeeze(-1)
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
    def emit_doc_cnn(self, doc_embeds, question_embeds, q_conv_res_trigram, q_weights):
        conv_res            = self.apply_context_convolution(doc_embeds, self.trigram_conv_1, self.trigram_conv_activation_1)
        conv_res            = self.apply_context_convolution(conv_res,   self.trigram_conv_2, self.trigram_conv_activation_2)
        sim_insens          = self.my_cosine_sim(question_embeds, doc_embeds).squeeze(0)
        sim_oh              = (sim_insens > (1 - (1e-3))).float()
        sim_sens            = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
        insensitive_pooled  = self.pooling_method(sim_insens)
        sensitive_pooled    = self.pooling_method(sim_sens)
        oh_pooled           = self.pooling_method(sim_oh)
        doc_emit            = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
        doc_emit            = doc_emit.unsqueeze(-1)
        return doc_emit
    def emit_doc_bigru(self, doc_embeds, question_embeds, q_conv_res_trigram, q_weights):
        conv_res, hn        = self.apply_context_gru(doc_embeds, self.context_h0)
        sim_insens          = self.my_cosine_sim(question_embeds, doc_embeds).squeeze(0)
        sim_oh              = (sim_insens > (1 - (1e-3))).float()
        sim_sens            = self.my_cosine_sim(q_conv_res_trigram, conv_res).squeeze(0)
        insensitive_pooled  = self.pooling_method(sim_insens)
        sensitive_pooled    = self.pooling_method(sim_sens)
        oh_pooled           = self.pooling_method(sim_oh)
        doc_emit            = self.get_output([oh_pooled, insensitive_pooled, sensitive_pooled], q_weights)
        doc_emit            = doc_emit.unsqueeze(-1)
        return doc_emit
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
    def emit_one(self, doc1_embeds, question_embeds, q_idfs, doc_gaf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        doc1_embeds         = autograd.Variable(torch.FloatTensor(doc1_embeds),         requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        # HANDLE QUESTION
        q_context           = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context           = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        # HANDLE DOCS
        good_out            = self.emit_doc_cnn(doc1_embeds, question_embeds, q_context, q_weights)
        #
        good_out_pp         = torch.cat([good_out, doc_gaf], -1)
        #
        final_good_output   = self.final_layer(good_out_pp)
        return final_good_output
    def forward(self, doc1_embeds, doc2_embeds, question_embeds, q_idfs, doc_gaf, doc_baf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        doc_baf             = autograd.Variable(torch.FloatTensor(doc_baf),             requires_grad=False)
        doc1_embeds         = autograd.Variable(torch.FloatTensor(doc1_embeds),         requires_grad=False)
        doc2_embeds         = autograd.Variable(torch.FloatTensor(doc2_embeds),         requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        doc_baf             = autograd.Variable(torch.FloatTensor(doc_baf),             requires_grad=False)
        # HANDLE QUESTION
        q_context           = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context           = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        # HANDLE DOCS
        good_out            = self.emit_doc_cnn(doc1_embeds, question_embeds, q_context, q_weights)
        bad_out             = self.emit_doc_cnn(doc2_embeds, question_embeds, q_context, q_weights)
        #
        good_out_pp         = torch.cat([good_out, doc_gaf], -1)
        bad_out_pp          = torch.cat([bad_out, doc_baf], -1)
        #
        final_good_output   = self.final_layer(good_out_pp)
        final_bad_output    = self.final_layer(bad_out_pp)
        #
        loss1               = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output

##################

model = Sent_Posit_Drmm_Modeler()
resume_from = '/home/dpappas/best_checkpoint.pth.tar'
load_model_from_checkpoint(resume_from)

##################

print('Loading Data')
dataloc = '/home/dpappas/PycharmProjects/aueb-bioasq6-master/bioasq6_data/'
with open(dataloc + 'test_batch_1/bioasq6_bm25_top100.test.pkl', 'rb') as f:
  data = pickle.load(f)
with open(dataloc + 'test_batch_1/bioasq6_bm25_docset_top100.test.pkl', 'rb') as f:
  docs = pickle.load(f)

words = {}
GetWords(data, docs, words)

##################

print('Making preds')
json_preds = {}
json_preds['questions'] = []
num_docs = 0
for i in range(len(data['queries'])):
  num_docs += 1
  dy.renew_cg()
  #########
  qtext = data['queries'][i]['query_text']
  qwds, qvecs, qconv = model.MakeInputs(qtext)
  #########
  rel_scores = {}
  for j in range(len(data['queries'][i]['retrieved_documents'])):
    doc_id              = data['queries'][i]['retrieved_documents'][j]['doc_id']
    dtext               = (docs[doc_id]['title'] + ' <title> ' + docs[doc_id]['abstractText'])
    dwds, dvecs, dconv  = model.MakeInputs(dtext)
    bm25                = data['queries'][i]['retrieved_documents'][j]['norm_bm25_score']
    efeats              = model.GetExtraFeatures(qtext, dtext, bm25)
    efeats_vec          = dy.inputVector(efeats)
    score               = model.GetQDScore(qwds, qconv, dwds, dconv, efeats_vec)
    rel_scores[j]       = score.value()
  #########
  top = heapq.nlargest(10, rel_scores, key=rel_scores.get)
  utils.JsonPredsAppend(json_preds, data, i, top)
  dy.renew_cg()

utils.DumpJson(json_preds, 'abel_test_preds_batch' + sys.argv[2] + '.json')
print('Done')


