
import sys
import math
import  os
import  re
import  numpy as np
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  pickle
import  torch.autograd as autograd
from    gensim.models.keyedvectors import KeyedVectors
from    nltk.tokenize import sent_tokenize
from    difflib import SequenceMatcher
from    pprint import pprint

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def tokenize(x):
  return bioclean(x)

def idf_val(w):
    if w in idf:
        return idf[w]
    return max_idf

def get_words(s):
    sl  = tokenize(s)
    sl  = [s for s in sl]
    sl2 = [s for s in sl if idf_val(s) >= 2.0]
    return sl, sl2

def get_embeds(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if(tok in wv):
            ret1.append(tok)
            ret2.append(wv[tok])
    return ret1, np.array(ret2, 'float64')

def load_idfs(idf_path):
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    ret = {}
    for w in idf:
        ret[w] = idf[w]
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    idf = None
    print('Loaded idf tables with max idf {}'.format(max_idf))
    return ret, max_idf

def uwords(words):
  uw = {}
  for w in words:
    uw[w] = 1
  return [w for w in uw]

def ubigrams(words):
  uw = {}
  prevw = "<pw>"
  for w in words:
    uw[prevw + '_' + w] = 1
    prevw = w
  return [w for w in uw]

def query_doc_overlap(qwords, dwords):
    # % Query words in doc.
    qwords_in_doc = 0
    idf_qwords_in_doc = 0.0
    idf_qwords = 0.0
    for qword in uwords(qwords):
      idf_qwords += idf_val(qword)
      for dword in uwords(dwords):
        if qword == dword:
          idf_qwords_in_doc += idf_val(qword)
          qwords_in_doc     += 1
          break
    if len(qwords) <= 0:
      qwords_in_doc_val = 0.0
    else:
      qwords_in_doc_val = (float(qwords_in_doc) /
                           float(len(uwords(qwords))))
    if idf_qwords <= 0.0:
      idf_qwords_in_doc_val = 0.0
    else:
      idf_qwords_in_doc_val = float(idf_qwords_in_doc) / float(idf_qwords)
    # % Query bigrams  in doc.
    qwords_bigrams_in_doc = 0
    idf_qwords_bigrams_in_doc = 0.0
    idf_bigrams = 0.0
    for qword in ubigrams(qwords):
      wrds = qword.split('_')
      idf_bigrams += idf_val(wrds[0]) * idf_val(wrds[1])
      for dword in ubigrams(dwords):
        if qword == dword:
          qwords_bigrams_in_doc += 1
          idf_qwords_bigrams_in_doc += (idf_val(wrds[0]) * idf_val(wrds[1]))
          break
    if len(qwords) <= 0:
      qwords_bigrams_in_doc_val = 0.0
    else:
      qwords_bigrams_in_doc_val = (float(qwords_bigrams_in_doc) / float(len(ubigrams(qwords))))
    if idf_bigrams <= 0.0:
      idf_qwords_bigrams_in_doc_val = 0.0
    else:
      idf_qwords_bigrams_in_doc_val = (float(idf_qwords_bigrams_in_doc) / float(idf_bigrams))
    return [qwords_in_doc_val,
            qwords_bigrams_in_doc_val,
            idf_qwords_in_doc_val,
            idf_qwords_bigrams_in_doc_val]

def GetScores(qtext, dtext, bm25):
    qwords, qw2 = get_words(qtext)
    dwords, dw2 = get_words(dtext)
    qd1         = query_doc_overlap(qwords, dwords)
    bm25        = [bm25]
    return qd1[0:3] + bm25

def prep_data(quest, good_doc_text, good_mesh, the_bm25=7.45):
    quest_tokens, quest_embeds  = get_embeds(tokenize(quest), wv)
    q_idfs                      = np.array([[idf_val(qw)] for qw in quest_tokens], 'float')
    good_doc_af                 = GetScores(quest, good_doc_text, the_bm25)
    good_sents                  = sent_tokenize(good_doc_text)
    good_sents_embeds, good_sents_escores, held_out_sents = [], [], []
    for good_text in good_sents:
        good_tokens, good_embeds = get_embeds(tokenize(good_text), wv)
        good_escores = GetScores(quest, good_text, the_bm25)[:-1]
        if (len(good_embeds) > 0):
            good_sents_embeds.append(good_embeds)
            good_sents_escores.append(good_escores)
            held_out_sents.append(good_text)
    good_mesh_embeds    = []
    held_out_mesh       = []
    for gm in ['mgmx']+good_mesh:
        mtoks, membs = get_embeds(gm.lower().split(), wv)
        if(len(mtoks)>0):
            good_mesh_embeds.append(membs)
            held_out_mesh.append(gm)
    return good_sents, good_sents_embeds, good_sents_escores, good_doc_af, good_mesh_embeds, held_out_mesh, held_out_sents, quest_tokens, quest_embeds, q_idfs

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

def load_model_from_checkpoint(resume_from):
    global start_epoch, optimizer
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

def get_one_output(quest, good_doc_text, good_meshes):
    (
        good_sents,         good_sents_embeds,
        good_sents_escores, good_doc_af,
        good_mesh_embeds,   held_out_mesh,
        held_out_sents,     quest_tokens,
        quest_embeds,       q_idfs
    ) = prep_data(quest, good_doc_text, good_meshes, the_bm25=7.45)
    doc_emit_, gs_emits_ = model.emit_one(

        doc1_sents_embeds=good_sents_embeds,
        question_embeds=quest_embeds,
        q_idfs=q_idfs,
        sents_gaf=good_sents_escores,
        doc_gaf=good_doc_af
    )
    # emition                 = doc_emit_.cpu().item()
    emitss                  = gs_emits_.tolist()
    sent_ret                = [
        (
            sent,
            int(math.floor(emitss[held_out_sents.index(sent)]*100)) if(sent in held_out_sents) else 0
        )
        for sent in good_sents
    ]
    #
    return sent_ret

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self,
             embedding_dim          = 30,
             k_for_maxpool          = 5,
             sentence_out_method    = 'MLP',
             k_sent_maxpool         = 1
         ):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.k                                      = k_for_maxpool
        self.k_sent_maxpool                         = k_sent_maxpool
        self.doc_add_feats                          = 11
        self.sent_add_feats                         = 10
        #
        self.embedding_dim                          = embedding_dim
        self.sentence_out_method                    = sentence_out_method
        # to create q weights
        self.init_context_module()
        self.init_question_weight_module()
        self.init_mlps_for_pooled_attention()
        self.init_sent_output_layer()
        self.init_doc_out_layer()
        # doc loss func
        self.margin_loss        = nn.MarginRankingLoss(margin=1.0)
        if(use_cuda):
            self.margin_loss    = self.margin_loss.cuda()
    def init_mesh_module(self):
        self.mesh_h0    = autograd.Variable(torch.randn(1, 1, self.embedding_dim))
        self.mesh_gru   = nn.GRU(self.embedding_dim, self.embedding_dim)
        if(use_cuda):
            self.mesh_h0    = self.mesh_h0.cuda()
            self.mesh_gru   = self.mesh_gru.cuda()
    def init_context_module(self):
        self.trigram_conv_1             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        # self.trigram_conv_activation_1  = torch.nn.LeakyReLU(negative_slope=0.1)
        self.trigram_conv_activation_1 = torch.nn.Sigmoid()
        self.trigram_conv_2             = nn.Conv1d(self.embedding_dim, self.embedding_dim, 3, padding=2, bias=True)
        # self.trigram_conv_activation_2  = torch.nn.LeakyReLU(negative_slope=0.1)
        self.trigram_conv_activation_2 = torch.nn.Sigmoid()
        if(use_cuda):
            self.trigram_conv_1             = self.trigram_conv_1.cuda()
            self.trigram_conv_2             = self.trigram_conv_2.cuda()
            self.trigram_conv_activation_1  = self.trigram_conv_activation_1.cuda()
            self.trigram_conv_activation_2  = self.trigram_conv_activation_2.cuda()
    def init_question_weight_module(self):
        self.q_weights_mlp      = nn.Linear(self.embedding_dim+1, 1, bias=True)
        if(use_cuda):
            self.q_weights_mlp  = self.q_weights_mlp.cuda()
    def init_mlps_for_pooled_attention(self):
        self.linear_per_q1      = nn.Linear(3 * 3, 8, bias=True)
        self.my_relu1           = torch.nn.LeakyReLU(negative_slope=0.1)
        self.linear_per_q2      = nn.Linear(8, 1, bias=True)
        if(use_cuda):
            self.linear_per_q1  = self.linear_per_q1.cuda()
            self.linear_per_q2  = self.linear_per_q2.cuda()
            self.my_relu1       = self.my_relu1.cuda()
    def init_sent_output_layer(self):
        if(self.sentence_out_method == 'MLP'):
            self.sent_out_layer_1       = nn.Linear(self.sent_add_feats+1, 8, bias=False)
            self.sent_out_activ_1       = torch.nn.LeakyReLU(negative_slope=0.1)
            self.sent_out_layer_2       = nn.Linear(8, 1, bias=False)
            if(use_cuda):
                self.sent_out_layer_1   = self.sent_out_layer_1.cuda()
                self.sent_out_activ_1   = self.sent_out_activ_1.cuda()
                self.sent_out_layer_2   = self.sent_out_layer_2.cuda()
        else:
            self.sent_res_h0    = autograd.Variable(torch.randn(2, 1, 5))
            self.sent_res_bigru = nn.GRU(input_size=self.sent_add_feats+1, hidden_size=5, bidirectional=True, batch_first=False)
            self.sent_res_mlp   = nn.Linear(10, 1, bias=False)
            if(use_cuda):
                self.sent_res_h0    = self.sent_res_h0.cuda()
                self.sent_res_bigru = self.sent_res_bigru.cuda()
                self.sent_res_mlp   = self.sent_res_mlp.cuda()
    def init_doc_out_layer(self):
        self.final_layer_1 = nn.Linear(self.doc_add_feats+self.k_sent_maxpool, 8, bias=True)
        self.final_activ_1  = torch.nn.LeakyReLU(negative_slope=0.1)
        self.final_layer_2  = nn.Linear(8, 1, bias=True)
        self.oo_layer       = nn.Linear(2, 1, bias=True)
        if(use_cuda):
            self.final_layer_1  = self.final_layer_1.cuda()
            self.final_activ_1  = self.final_activ_1.cuda()
            self.final_layer_2  = self.final_layer_2.cuda()
            self.oo_layer       = self.oo_layer.cuda()
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
        # residual
        conv_res = conv_res + the_input
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
    def do_for_one_doc_cnn(self, doc_sents_embeds, sents_af, question_embeds, q_conv_res_trigram, q_weights, k2):
        res = []
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            if(use_cuda):
                sent_embeds     = sent_embeds.cuda()
                gaf             = gaf.cuda()
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
        hn  = self.context_h0
        for i in range(len(doc_sents_embeds)):
            sent_embeds         = autograd.Variable(torch.FloatTensor(doc_sents_embeds[i]), requires_grad=False)
            gaf                 = autograd.Variable(torch.FloatTensor(sents_af[i]), requires_grad=False)
            if(use_cuda):
                sent_embeds     = sent_embeds.cuda()
                gaf             = gaf.cuda()
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
        res     = torch.sort(res,0)[0]
        res     = res[-k:].squeeze(-1)
        if(len(res.size())==0):
            res = res.unsqueeze(0)
        if(res.size()[0] < k):
            to_concat       = torch.zeros(k - res.size()[0])
            if(use_cuda):
                to_concat   = to_concat.cuda()
            res             = torch.cat([res, to_concat], -1)
        return res
    def get_max_and_average_of_k_max(self, res, k):
        k_max_pooled            = self.get_kmax(res, k)
        average_k_max_pooled    = k_max_pooled.sum()/float(k)
        the_maximum             = k_max_pooled[-1]
        the_concatenation       = torch.cat([the_maximum, average_k_max_pooled.unsqueeze(0)])
        return the_concatenation
    def get_average(self, res):
        res = torch.sum(res) / float(res.size()[0])
        return res
    def get_maxmin_max(self, res):
        res = self.min_max_norm(res)
        res = torch.max(res)
        return res
    def apply_mesh_gru(self, mesh_embeds):
        mesh_embeds             = autograd.Variable(torch.FloatTensor(mesh_embeds), requires_grad=False)
        if(use_cuda):
            mesh_embeds         = mesh_embeds.cuda()
        output, hn              = self.mesh_gru(mesh_embeds.unsqueeze(1), self.mesh_h0)
        return output[-1,0,:]
    def get_mesh_rep(self, meshes_embeds, q_context):
        meshes_embeds   = [self.apply_mesh_gru(mesh_embeds) for mesh_embeds in meshes_embeds]
        meshes_embeds   = torch.stack(meshes_embeds)
        sim_matrix      = self.my_cosine_sim(meshes_embeds, q_context).squeeze(0)
        max_sim         = torch.sort(sim_matrix, -1)[0][:, -1]
        output          = torch.mm(max_sim.unsqueeze(0), meshes_embeds)[0]
        return output
    def emit_one(self, doc1_sents_embeds, question_embeds, q_idfs, sents_gaf, doc_gaf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        if(use_cuda):
            q_idfs          = q_idfs.cuda()
            question_embeds = question_embeds.cuda()
            doc_gaf         = doc_gaf.cuda()
        #
        q_context           = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context           = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits  = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights, self.k_sent_maxpool)
        #
        good_out_pp         = torch.cat([good_out, doc_gaf], -1)
        #
        final_good_output   = self.final_layer_1(good_out_pp)
        final_good_output   = self.final_activ_1(final_good_output)
        final_good_output   = self.final_layer_2(final_good_output)
        #
        gs_emits            = gs_emits.unsqueeze(-1)
        gs_emits            = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits            = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits            = torch.sigmoid(gs_emits)
        #
        return final_good_output, gs_emits
    def forward(self, doc1_sents_embeds, doc2_sents_embeds, question_embeds, q_idfs, sents_gaf, sents_baf, doc_gaf, doc_baf):
        q_idfs              = autograd.Variable(torch.FloatTensor(q_idfs),              requires_grad=False)
        question_embeds     = autograd.Variable(torch.FloatTensor(question_embeds),     requires_grad=False)
        doc_gaf             = autograd.Variable(torch.FloatTensor(doc_gaf),             requires_grad=False)
        doc_baf             = autograd.Variable(torch.FloatTensor(doc_baf),             requires_grad=False)
        if(use_cuda):
            q_idfs          = q_idfs.cuda()
            question_embeds = question_embeds.cuda()
            doc_gaf         = doc_gaf.cuda()
            doc_baf         = doc_baf.cuda()
        #
        q_context           = self.apply_context_convolution(question_embeds,   self.trigram_conv_1, self.trigram_conv_activation_1)
        q_context           = self.apply_context_convolution(q_context,         self.trigram_conv_2, self.trigram_conv_activation_2)
        #
        q_weights           = torch.cat([q_context, q_idfs], -1)
        q_weights           = self.q_weights_mlp(q_weights).squeeze(-1)
        q_weights           = F.softmax(q_weights, dim=-1)
        #
        good_out, gs_emits  = self.do_for_one_doc_cnn(doc1_sents_embeds, sents_gaf, question_embeds, q_context, q_weights, self.k_sent_maxpool)
        bad_out, bs_emits   = self.do_for_one_doc_cnn(doc2_sents_embeds, sents_baf, question_embeds, q_context, q_weights, self.k_sent_maxpool)
        #
        good_out_pp         = torch.cat([good_out, doc_gaf], -1)
        bad_out_pp          = torch.cat([bad_out, doc_baf], -1)
        #
        final_good_output   = self.final_layer_1(good_out_pp)
        final_good_output   = self.final_activ_1(final_good_output)
        final_good_output   = self.final_layer_2(final_good_output)
        #
        gs_emits            = gs_emits.unsqueeze(-1)
        gs_emits            = torch.cat([gs_emits, final_good_output.unsqueeze(-1).expand_as(gs_emits)], -1)
        gs_emits            = self.oo_layer(gs_emits).squeeze(-1)
        gs_emits            = torch.sigmoid(gs_emits)
        #
        final_bad_output    = self.final_layer_1(bad_out_pp)
        final_bad_output    = self.final_activ_1(final_bad_output)
        final_bad_output    = self.final_layer_2(final_bad_output)
        #
        bs_emits            = bs_emits.unsqueeze(-1)
        bs_emits            = torch.cat([bs_emits, final_good_output.unsqueeze(-1).expand_as(bs_emits)], -1)
        bs_emits            = self.oo_layer(bs_emits).squeeze(-1)
        bs_emits            = torch.sigmoid(bs_emits)
        #
        loss1               = self.my_hinge_loss(final_good_output, final_bad_output)
        return loss1, final_good_output, final_bad_output, gs_emits, bs_emits

w2v_bin_path        = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
idf_pickle_path     = '/home/dpappas/bioasq_all/idf.pkl'
resume_from         = '/media/dpappas/dpappas_data/models_out/sigir_joint_simple_2L_only_positive_sents_0p01_run_0/best_checkpoint.pth.tar'
# resume_from         = './best_checkpoint.pth.tar'

idf, max_idf        = load_idfs(idf_pickle_path)
print('loading w2v')
wv = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
wv = dict([(word, wv[word]) for word in wv.vocab.keys()])

k_for_maxpool   = 5
k_sent_maxpool  = 2
embedding_dim   = 30 #200
lr              = 0.01
b_size          = 32
max_epoch       = 10

use_cuda    = False
model       = Sent_Posit_Drmm_Modeler(embedding_dim=embedding_dim, k_for_maxpool=k_for_maxpool)

# resume_from = '/home/dpappas/model_18_run_3/best_checkpoint.pth.tar'
load_model_from_checkpoint(resume_from)
print('LOADED model')

# quest                       = 'What is my name?'
# good_doc_text               = 'my name is tilalop'
# good_meshes                 = ['A B', 'C']

from colour import Color
from flask import Flask
from flask import render_template
from flask import request
import random

red     = Color("white")
colors  = list(red.range_to(Color("yellow"), 100))
colors  = [c.get_hex_l() for c in colors]

app = Flask(__name__)

@app.route("/")
def get_news():
    return render_template("home.html")

@app.route("/submit_question", methods=["POST"])
def get_quest_results():
    q = request.form.get("the_quest").strip()
    d = request.form.get("the_doc").strip()
    m = request.form.get("the_mesh").strip()
    m = [t.strip() for t in m.split('||')]
    sent_res = get_one_output(q, d, m)
    # return render_template("home.html")
    # add the question as header
    ret_html = '<h2>Question: </h2>'
    ret_html += '<p>{}</p>'.format(q)
    ret_html += '</br></br>'
    # add the scored sentences
    ret_html += '<h2>Document:</h2>'
    for res in sent_res:
        score   = res[1] if (res[1]>0) else 0
        sent    = res[0]
        ret_html += '<div score="{}" style="background-color:{}">{}</div>'.format(score, colors[score], sent)
    ret_html += '</br></br>'
    # add the scored mesh terms
    ret_html += '<h2>Mesh Terms:</h2>'
    mesh_res = []
    for res in mesh_res:
        score   = res[1] if (res[1]>0) else 0
        sent    = res[0]
        ret_html += '<div score="{}" style="background-color:{}">{}</div>'.format(score, colors[score], sent)
    ret_html += '</br></br>'
    ret_html += '<p>Extracted from model: {}</p>'.format(resume_from)
    #
    return ret_html

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

'''
In what substances is Papilin largely resistant ?

A sulfated glycoprotein was isolated from the culture media of Drosophila Kc cells and named papilin. Affinity purified antibodies against this protein localized it primarily to the basement membranes of embryos. The antibodies cross-reacted with another material which was not sulfated and appeared to be the core protein of papilin, which is proteoglycan-like. After reduction, papilin electrophoresed in sodium dodecyl sulfate-polyacrylamide gel electrophoresis as a broad band of about 900,000 apparent molecular weight and the core protein as a narrow band of approximately 400,000. The core protein was formed by some cell lines and by other cells on incubation with 1 mM 4-methylumbelliferyl xyloside, which inhibited formation of the proteoglycan-like form. The buoyant density of papilin in CsCl/4 M guanidine hydrochloride is 1.4 g/ml, that of the core protein is much less. Papilin forms oligomers linked by disulfide bridges, as shown by sodium dodecyl sulfate-agarose gel electrophoresis and electron microscopy. The protomer is a 225 +/- 15-nm thread which is disulfide-linked into a loop with fine, protruding thread ends. Oligomers form clover-leaf-like structures. The protein contains 22% combined serine and threonine residues and 25% combined aspartic and glutamic residues. 10 g of polypeptide has attached 6.4 g of glucosamine, 3.1 g of galactosamine, 6.1 g of uronic acid, and 2.7 g of neutral sugars. There are about 80 O-linked carbohydrate chains/core protein molecule. Sulfate is attached to these chains. The O-linkage is through an unidentified neutral sugar. Papilin is largely resistant to common glycosidases and several proteases. The degree of sulfation varies with the sulfate concentration of the incubation medium. This proteoglycan-like glycoprotein differs substantially from corresponding proteoglycans found in vertebrate basement membranes, in contrast to Drosophila basement membrane laminin and collagen IV which have been conserved evolutionarily.

Amino Acids/analysis || Animals || Basement Membrane/analysis || Carbohydrates/analysis || Drosophila/analysis* || Drosophila Proteins* || Electrophoresis, Polyacrylamide Gel || Fluorescent Antibody Technique || Glycoproteins/analysis* || Microscopy, Electron || Molecular Weight || Proteoglycans/analysis* || Sulfates/analysis

'''
