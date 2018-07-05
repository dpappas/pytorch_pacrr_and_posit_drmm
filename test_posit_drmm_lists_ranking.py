
import os
import re
import sys
import random
import numpy as np
import cPickle as pickle
from pprint import pprint
from nltk.tokenize import sent_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import json
from tqdm import tqdm

k_for_maxpool   = 5
lr              = 0.01
reg_lambda      = 0.1

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

def load_model_from_checkpoint(resume_from):
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

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
        self.sent_filters_conv_1                    = torch.nn.Parameter(torch.randn(self.embedding_dim,1,3,self.embedding_dim))
        self.quest_filters_conv_1                   = self.sent_filters_conv_1
        self.sent_filters_conv_2                    = torch.nn.Parameter(torch.randn(self.embedding_dim,1,2,self.embedding_dim))
        self.quest_filters_conv_2                   = self.sent_filters_conv_2
        #
        self.linear_per_q1                          = nn.Linear(8, 8, bias=True)
        self.linear_per_q2                          = nn.Linear(8, 1, bias=True)
        self.my_relu1                               = torch.nn.PReLU()
        self.my_relu2                               = torch.nn.PReLU()
        self.my_drop1                               = nn.Dropout(p=0.2)
        self.my_loss                                = nn.MarginRankingLoss(margin=0.9)
    def apply_convolution(self, the_input, the_filters):
        filter_size = the_filters.size(2)
        the_input   = the_input.unsqueeze(0)
        conv_res    = F.conv2d(the_input.unsqueeze(1), the_filters, bias=None, stride=1, padding=(int(filter_size/2)+1, 0))
        conv_res    = conv_res[:, :, -1*the_input.size(1):, :]
        conv_res    = conv_res.squeeze(-1).transpose(1,2)
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
        sorted_res              = torch.sort(sim_matrix, -1)[0]
        k_max_pooled            = sorted_res[:,-self.k:]
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)
        the_maximum             = k_max_pooled[:, -1]
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled], dim=-1)
        return the_concatenation
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
        lo      = self.my_relu2(lo)
        lo      = lo.squeeze(-1)
        sr      = lo.sum(-1) / lo.size(-1)
        return sr
    def forward(self, doc1, question, doc1_sim):
        #
        question                            = autograd.Variable(torch.LongTensor(question), requires_grad=False)
        doc1                                = autograd.Variable(torch.LongTensor(doc1), requires_grad=False)
        #
        question_embeds                     = self.word_embeddings(question)
        doc1_embeds                         = self.word_embeddings(doc1)
        #
        q_conv_res                          = self.apply_convolution(question_embeds,   self.quest_filters_conv_1)
        doc1_conv_1                         = self.apply_convolution(doc1_embeds,       self.sent_filters_conv_1)
        doc1_conv_2                         = self.apply_convolution(doc1_embeds,       self.sent_filters_conv_2)
        #
        similarity_insensitive_doc1         = self.my_cosine_sim(question_embeds, doc1_embeds).squeeze(0)
        similarity_insensitive_doc1         = self.apply_masks_on_similarity(doc1, question, similarity_insensitive_doc1)
        #
        similarity_sensitive_doc1_1         = self.my_cosine_sim(q_conv_res, doc1_conv_1).squeeze(0)
        similarity_sensitive_doc1_2         = self.my_cosine_sim(q_conv_res, doc1_conv_2).squeeze(0)
        #
        similarity_one_hot_doc1             = autograd.Variable(torch.FloatTensor(doc1_sim).transpose(0,1), requires_grad=False)
        #
        similarity_insensitive_pooled_doc1  = self.pooling_method(similarity_insensitive_doc1)
        similarity_sensitive_pooled_doc1_1  = self.pooling_method(similarity_sensitive_doc1_1)
        similarity_sensitive_pooled_doc1_2  = self.pooling_method(similarity_sensitive_doc1_2)
        similarity_one_hot_pooled_doc1      = self.pooling_method(similarity_one_hot_doc1)
        #
        doc1_emit = self.get_output(
            [
                similarity_one_hot_pooled_doc1,
                similarity_insensitive_pooled_doc1,
                similarity_sensitive_pooled_doc1_1,
                similarity_sensitive_pooled_doc1_2
            ]
        )
        return doc1_emit

print('Compiling model...')
matrix              = np.load('/home/dpappas/joint_task_list_batches/embedding_matrix.npy')
model               = Sent_Posit_Drmm_Modeler(pretrained_embeds=matrix, k_for_maxpool=k_for_maxpool)
params              = list(set(model.parameters()) - set([model.word_embeddings.weight]))

# resume_dir          = '/home/dpappas/posit_drmm_lists_rank/'
resume_dir          = '/home/dpappas/posit_drmm_lists_rank_3timesloop/'
resume_from         = resume_dir+'best_checkpoint.pth.tar'
load_model_from_checkpoint(resume_from)

token_to_index_f    = '/home/dpappas/joint_task_list_batches/t2i.p'
t2i                 = pickle.load(open(token_to_index_f,'rb'))
abs_path            = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.test.pkl'
all_abs             = pickle.load(open(abs_path,'rb'))
bm25_scores_path    = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.test.pkl'
bm25_scores         = pickle.load(open(bm25_scores_path, 'rb'))

data = {}
data['questions'] = []
for quer in tqdm(bm25_scores['queries']):
    dato    = {'body':quer['query_text'], 'id':quer['query_id'], 'documents':[]}
    doc_res = {}
    for retr in quer['retrieved_documents']:
        doc_id      = retr['doc_id']
        passage     = all_abs[doc_id]['title'] + ' ' + all_abs[doc_id]['abstractText']
        all_sims    = get_sim_mat(bioclean(passage), bioclean(quer['query_text']))
        sents_inds  = [get_index(token, t2i) for token in bioclean(passage)]
        quest_inds  = [get_index(token, t2i) for token in bioclean(quer['query_text'])]
        doc1_emit_  = model(doc1=sents_inds, question=quest_inds, doc1_sim=all_sims)
        print doc1_emit_
        doc_res[doc_id] = float(doc1_emit_)
    doc_res             = sorted(doc_res.keys(), key=lambda x: doc_res[x], reverse=True)
    doc_res             = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm) for pm in doc_res[:100]]
    dato['documents']   = doc_res
    data['questions'].append(dato)

with open(resume_dir+'elk_relevant_abs_posit_drmm_lists.json', 'w') as f:
    f.write(json.dumps(data, indent=4, sort_keys=True))

'''
python /home/DATA/Biomedical/document_ranking/eval/run_eval.py \
/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq.test.json \
/home/dpappas/posit_drmm_lists_rank/elk_relevant_abs_posit_drmm_lists.json

python /home/DATA/Biomedical/document_ranking/eval/run_eval.py \
/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq.test.json \
/home/dpappas/posit_drmm_lists_rank_3timesloop/elk_relevant_abs_posit_drmm_lists.json
'''
