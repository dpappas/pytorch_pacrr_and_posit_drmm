
import re
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

def get_index(token, t2i):
    try:
        return t2i[token]
    except KeyError:
        return t2i['UNKN']

def first_alpha_is_upper(sent):
    specials = [
        '__EU__','__SU__','__EMS__','__SMS__','__SI__',
        '__ESB','__SSB__','__EB__','__SB__','__EI__',
        '__EA__','__SA__','__SQ__','__EQ__','__EXTLINK',
        '__XREF','__URI', '__EMAIL','__ARRAY','__TABLE',
        '__FIG','__AWID','__FUNDS'
    ]
    for special in specials:
        sent = sent.replace(special,'')
    for c in sent:
        if(c.isalpha()):
            if(c.isupper()):
                return True
            else:
                return False
    return False

def ends_with_special(sent):
    sent = sent.lower()
    ind = [item.end() for item in re.finditer('[\W\s]sp.|[\W\s]nos.|[\W\s]figs.|[\W\s]sp.[\W\s]no.|[\W\s][vols.|[\W\s]cv.|[\W\s]fig.|[\W\s]e.g.|[\W\s]et[\W\s]al.|[\W\s]i.e.|[\W\s]p.p.m.|[\W\s]cf.|[\W\s]n.a.', sent)]
    if(len(ind)==0):
        return False
    else:
        ind = max(ind)
        if (len(sent) == ind):
            return True
        else:
            return False

def split_sentences(text):
    sents = [l.strip() for l in sent_tokenize(text)]
    ret = []
    i = 0
    while (i < len(sents)):
        sent = sents[i]
        while (
            ((i + 1) < len(sents)) and
            (
                ends_with_special(sent) or
                not first_alpha_is_upper(sents[i+1].strip())
                # sent[-5:].count('.') > 1       or
                # sents[i+1][:10].count('.')>1   or
                # len(sent.split()) < 2          or
                # len(sents[i+1].split()) < 2
            )
        ):
            sent += ' ' + sents[i + 1]
            i += 1
        ret.append(sent.replace('\n',' ').strip())
        i += 1
    return ret

def get_sents(ntext):
    if(len(ntext.strip())>0):
        sents = []
        for subtext in ntext.split('\n'):
            subtext = re.sub( '\s+', ' ', subtext.replace('\n',' ') ).strip()
            if (len(subtext) > 0):
                ss = split_sentences(subtext)
                sents.extend([ s for s in ss if(len(s.strip())>0)])
        if(len(sents[-1]) == 0 ):
            sents = sents[:-1]
        return sents
    else:
        return []

def get_sim_mat(stoks, qtoks):
    sm = np.zeros((len(stoks), len(qtoks)))
    for i in range(len(qtoks)):
        for j in range(len(stoks)):
            if(qtoks[i] == stoks[j]):
                sm[j,i] = 1.
    return sm

def get_item_inds(item, question, t2i):
    doc_title   = get_sents(item['title'])
    doc_text    = get_sents(item['abstractText'])
    all_sents   = doc_title + doc_text
    all_sents   = [s for s in all_sents if(len(bioclean(s))>0)]
    #
    all_sims    = [get_sim_mat(bioclean(stoks), bioclean(question)) for stoks in all_sents]
    #
    sents_inds  = [[get_index(token, t2i) for token in bioclean(s)] for s in all_sents]
    quest_inds  = [get_index(token, t2i) for token in bioclean(question)]
    #
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

def data_yielder(bm25_scores, all_abs, t2i):
    for quer in bm25_scores[u'queries']:
        quest = quer['query_text']
        ret_pmids   = [t[u'doc_id'] for t in quer[u'retrieved_documents']]
        good_pmids  = [t for t in ret_pmids if t in quer[u'relevant_documents']]
        bad_pmids   = [t for t in ret_pmids if t not in quer[u'relevant_documents']]
        for gid in good_pmids:
            bid = random.choice(bad_pmids)
            good_sents_inds, good_quest_inds, good_all_sims = get_item_inds(all_abs[gid], quest, t2i)
            bad_sents_inds, bad_quest_inds, bad_all_sims    = get_item_inds(all_abs[bid], quest, t2i)
            yield good_sents_inds, good_all_sims, bad_sents_inds, bad_all_sims, bad_quest_inds

def dummy_test():
    good_sents_inds     = np.random.randint(0,100, (10,3))
    good_all_sims       = np.zeros((10,3, 4))
    bad_sents_inds      = np.random.randint(0,100, (7,5))
    bad_all_sims        = np.zeros((7, 5, 4))
    bad_quest_inds      = np.random.randint(0,100,(4))
    for epoch in range(200):
        optimizer.zero_grad()
        cost_, sent_ems, doc_ems = model(
            doc1_sents  = good_sents_inds,
            doc2_sents  = bad_sents_inds,
            question    = bad_quest_inds,
            doc1_sim    = good_all_sims,
            doc2_sim    = bad_all_sims
        )
        cost_.backward()
        optimizer.step()
        the_cost = cost_.cpu().item()
        print(the_cost)
    print(20 * '-')

def compute_the_cost(costs):
    cost_ = torch.stack(costs)
    cost_ = cost_.sum() / (1.0 * cost_.size(0))
    cost_.backward()
    optimizer.step()
    the_cost = cost_.cpu().item()
    print(the_cost)
    optimizer.zero_grad()

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

class Sent_Posit_Drmm_Modeler(nn.Module):
    def __init__(self, nof_filters, filters_size, pretrained_embeds, k_for_maxpool):
        super(Sent_Posit_Drmm_Modeler, self).__init__()
        self.nof_sent_filters                       = nof_filters           # number of filters for the convolution of sentences
        self.sent_filters_size                      = filters_size          # The size of the ngram filters we will apply on sentences
        self.nof_quest_filters                      = nof_filters           # number of filters for the convolution of the question
        self.quest_filters_size                     = filters_size          # The size of the ngram filters we will apply on question
        self.k                                      = k_for_maxpool         # k is for the average k pooling
        self.vocab_size                             = pretrained_embeds.shape[0]
        self.embedding_dim                          = pretrained_embeds.shape[1]
        self.word_embeddings                        = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeds))
        self.word_embeddings.weight.requires_grad   = False
        self.sent_filters_conv                      = torch.nn.Parameter(torch.randn(self.nof_sent_filters,1,self.sent_filters_size,self.embedding_dim))
        self.quest_filters_conv                     = self.sent_filters_conv
        self.linear_per_q                           = nn.Linear(6, 1, bias=True)
        self.my_loss                                = nn.MarginRankingLoss(margin=0.9)
    def apply_convolution(self, the_input, the_filters):
        filter_size = the_filters.size(2)
        the_input   = the_input.unsqueeze(0)
        conv_res    = F.conv2d(the_input.unsqueeze(1), the_filters, bias=None, stride=1, padding=(int(filter_size/2)+1, 0))
        conv_res    = conv_res[:, :, -1*the_input.size(1):, :]
        conv_res    = conv_res.squeeze(-1).transpose(1,2)
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
    def my_cosine_sim_many(self, quest, sents):
        ret = []
        for sent in sents:
            ret.append(self.my_cosine_sim(quest,sent).squeeze(0))
        return ret
    def pooling_method(self, sim_matrix):
        sorted_res              = torch.sort(sim_matrix, -1)[0]             # sort the input minimum to maximum
        k_max_pooled            = sorted_res[:,-self.k:]                    # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)        # average these k values
        the_maximum             = k_max_pooled[:, -1]                       # select the maximum value of each instance
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled], dim=-1) # concatenate maximum value and average of k-max values
        return the_concatenation     # return the concatenation
    def apply_masks_on_similarity(self, sentences, question, similarity):
        qq = (question > 1).float()
        for si in range(len(sentences)):
            ss              = (sentences[si] > 1).float()
            sim_mask1       = qq.unsqueeze(-1).expand_as(similarity[si])
            sim_mask2       = ss.unsqueeze(0).expand_as(similarity[si])
            similarity[si] *= sim_mask1
            similarity[si] *= sim_mask2
        return similarity
    def get_sent_output(self, similarity_one_hot_pooled, similarity_insensitive_pooled,similarity_sensitive_pooled):
        ret_r = []
        for j in range(len(similarity_one_hot_pooled)):
            temp = torch.cat([similarity_insensitive_pooled[j], similarity_sensitive_pooled[j], similarity_one_hot_pooled[j]], -1)
            lo = self.linear_per_q(temp).squeeze(-1)
            lo = F.sigmoid(lo)
            sr = lo.sum(-1) / lo.size(-1)
            ret_r.append(sr)
        return torch.stack(ret_r)
    def forward(self, doc1_sents, doc2_sents, question, doc1_sim, doc2_sim):
        #
        question     = autograd.Variable(torch.LongTensor(question), requires_grad=False)
        doc1_sents   = [autograd.Variable(torch.LongTensor(item), requires_grad=False) for item in doc1_sents]
        doc2_sents   = [autograd.Variable(torch.LongTensor(item), requires_grad=False) for item in doc2_sents]
        #
        question_embeds     = self.word_embeddings(question)
        doc1_sents_embeds   = [self.word_embeddings(sent) for sent in doc1_sents]
        doc2_sents_embeds   = [self.word_embeddings(sent) for sent in doc2_sents]
        #
        q_conv_res          = self.apply_convolution(question_embeds, self.quest_filters_conv)
        doc1_sents_conv     = [self.apply_convolution(sent, self.sent_filters_conv) for sent in doc1_sents_embeds]
        doc2_sents_conv     = [self.apply_convolution(sent, self.sent_filters_conv) for sent in doc2_sents_embeds]
        #
        similarity_insensitive_doc1 = self.my_cosine_sim_many(question_embeds, doc1_sents_embeds)
        similarity_insensitive_doc1 = self.apply_masks_on_similarity(doc1_sents, question, similarity_insensitive_doc1)
        similarity_insensitive_doc2 = self.my_cosine_sim_many(question_embeds, doc2_sents_embeds)
        similarity_insensitive_doc2 = self.apply_masks_on_similarity(doc2_sents, question, similarity_insensitive_doc2)
        #
        similarity_sensitive_doc1   = self.my_cosine_sim_many(q_conv_res, doc1_sents_conv)
        similarity_sensitive_doc2   = self.my_cosine_sim_many(q_conv_res, doc2_sents_conv)
        #
        similarity_one_hot_doc1     = [autograd.Variable(torch.FloatTensor(item).transpose(0,1), requires_grad=False) for item in doc1_sim]
        similarity_one_hot_doc2     = [autograd.Variable(torch.FloatTensor(item).transpose(0,1), requires_grad=False) for item in doc2_sim]
        #
        similarity_insensitive_pooled_doc1   = [self.pooling_method(item) for item in similarity_insensitive_doc1]
        similarity_sensitive_pooled_doc1     = [self.pooling_method(item) for item in similarity_sensitive_doc1]
        similarity_one_hot_pooled_doc1       = [self.pooling_method(item) for item in similarity_one_hot_doc1]
        #
        similarity_insensitive_pooled_doc2   = [self.pooling_method(item) for item in similarity_insensitive_doc2]
        similarity_sensitive_pooled_doc2     = [self.pooling_method(item) for item in similarity_sensitive_doc2]
        similarity_one_hot_pooled_doc2       = [self.pooling_method(item) for item in similarity_one_hot_doc2]
        #
        sent_output_doc1                     = self.get_sent_output(similarity_one_hot_pooled_doc1, similarity_insensitive_pooled_doc1, similarity_sensitive_pooled_doc1)
        sent_output_doc2                     = self.get_sent_output(similarity_one_hot_pooled_doc2, similarity_insensitive_pooled_doc2, similarity_sensitive_pooled_doc2)
        #
        doc1_emit   = sent_output_doc1.sum() / (1. * sent_output_doc1.size(0))
        doc2_emit   = sent_output_doc2.sum() / (1. * sent_output_doc2.size(0))
        #
        loss        = self.my_loss(doc1_emit.unsqueeze(0), doc2_emit.unsqueeze(0), torch.ones(1))
        # print(doc1_emit)
        # print(doc2_emit)
        # print(loss)
        return loss, doc1_emit, doc2_emit

print('LOADING embedding_matrix (14GB)...')
matrix          = np.load('/home/dpappas/joint_task_list_batches/embedding_matrix.npy')
# matrix          = np.random.random((290, 10))

nof_cnn_filters = 12
filters_size    = 3
k_for_maxpool   = 5
lr              = 0.1
print('Compiling model...')
model           = Sent_Posit_Drmm_Modeler(
    nof_filters=        nof_cnn_filters,
    filters_size=       filters_size,
    pretrained_embeds=  matrix,
    k_for_maxpool=      k_for_maxpool
)
params          = list(set(model.parameters()) - set([model.word_embeddings.weight]))
print_params(model)
del(matrix)
optimizer       = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# dummy_test()
# exit()

abs_path            = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.train.pkl'
bm25_scores_path    = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.train.pkl'
token_to_index_f    = '/home/dpappas/joint_task_list_batches/t2i.p'
print('Loading abs texts...')
all_abs             = pickle.load(open(abs_path,'rb'))
print('Loading retrieved docsc...')
bm25_scores         = pickle.load(open(bm25_scores_path, 'rb'))
print('Loading token to index files...')
t2i                 = pickle.load(open(token_to_index_f,'rb'))
print('Done')

dy = data_yielder(bm25_scores, all_abs, t2i)
bsize   = 64
optimizer.zero_grad()
for epoch in range(200):
    costs = []
    for good_sents_inds, good_all_sims, bad_sents_inds, bad_all_sims, quest_inds in dy:
        pass
        #
        instance_cost, sent_ems, doc_ems = model( good_sents_inds, bad_sents_inds, quest_inds, good_all_sims, bad_all_sims)
        costs.append(instance_cost)
        #
        if(len(costs) == bsize):
            compute_the_cost(costs)
            costs = []
    if(len(costs)>0):
        compute_the_cost(costs)
        costs = []
    print(20 * '-')


















