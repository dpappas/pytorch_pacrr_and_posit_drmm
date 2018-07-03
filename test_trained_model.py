
import cPickle as pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from nltk.tokenize import sent_tokenize
import numpy as np
import json
import os
import re
from tqdm import tqdm
from pprint import pprint

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

def loadGloveModel(w2v_voc, w2v_vec):
    '''
    :param w2v_voc: the txt file with the vocabulary extracted from gensim
    :param w2v_vec: the txt file with the vectors extracted from gensim
    :return: vocab is a python dictionary with the indices of each word. matrix is a numpy matrix with all the vectors.
             PAD is special token for padding to maximum length. It has a vector of zeros.
             UNK is special token for any token not found in the vocab. It has a vector equal to the average of all other vectors.
    '''
    temp_vocab  = pickle.load(open(w2v_voc,'rb'))
    temp_matrix = pickle.load(open(w2v_vec,'rb'))
    print("Loading Glove Model")
    vocab, matrix   = {}, []
    vocab['PAD']    = 0
    vocab['UNKN']   = len(vocab)
    for i in range(len(temp_vocab)):
        matrix.append(temp_matrix[i])
        vocab[temp_vocab[i]] = len(vocab)
    matrix          = np.array(matrix)
    av              = np.average(matrix,0)
    pad             = np.zeros(av.shape)
    matrix          = np.vstack([pad, av, matrix])
    print("Done.",len(vocab)," words loaded!")
    return vocab, matrix

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

def fix_relevant_snippets(relevant_parts):
    relevant_snippets = []
    for rt in relevant_parts:
        relevant_snippets.extend(get_sents(rt))
    return relevant_snippets

class Posit_Drmm_Modeler(nn.Module):
    def __init__(self, nof_filters, filters_size, pretrained_embeds, k_for_maxpool):
        super(Posit_Drmm_Modeler, self).__init__()
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
        self.sent_filters_conv  = torch.nn.Parameter(torch.randn(self.nof_sent_filters,1,self.sent_filters_size,self.embedding_dim))
        self.quest_filters_conv = self.sent_filters_conv
        self.linear_per_q       = nn.Linear(6, 1, bias=True)
        self.bce_loss           = torch.nn.BCELoss()
    def get_embeds(self, items):
        return [self.word_embeddings(item)for item in items]
    def apply_convolution(self, listed_inputs, the_filters):
        ret             = []
        filter_size     = the_filters.size(2)
        for the_input in listed_inputs:
            the_input   = the_input.unsqueeze(0)
            conv_res    = F.conv2d(the_input.unsqueeze(1), the_filters, bias=None, stride=1, padding=(int(filter_size/2)+1, 0))
            conv_res    = conv_res[:, :, -1*the_input.size(1):, :]
            conv_res    = conv_res.squeeze(-1).transpose(1,2)
            ret.append(conv_res.squeeze(0))
            # ret.append(conv_res)
        return ret
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
        k_max_pooled            = sorted_res[:,-self.k:]                # select the last k of each instance in our data
        average_k_max_pooled    = k_max_pooled.sum(-1)/float(self.k)        # average these k values
        the_maximum             = k_max_pooled[:, -1]                 # select the maximum value of each instance
        the_concatenation       = torch.stack([the_maximum, average_k_max_pooled], dim=-1) # concatenate maximum value and average of k-max values
        return the_concatenation     # return the concatenation
    def get_sent_output(self, similarity_one_hot_pooled, similarity_insensitive_pooled,similarity_sensitive_pooled):
        ret = []
        for bi in range(len(similarity_one_hot_pooled)):
            ret_r = []
            for j in range(len(similarity_one_hot_pooled[bi])):
                temp = torch.cat(
                    [
                        similarity_insensitive_pooled[bi][j],
                        similarity_sensitive_pooled[bi][j],
                        similarity_one_hot_pooled[bi][j]
                    ],
                    -1
                )
                # print(temp.size())
                lo = self.linear_per_q(temp).squeeze(-1)
                lo = F.sigmoid(lo)
                # lo =  F.hardtanh(lo, min_val=0, max_val=1)
                sr = lo.sum(-1) / lo.size(-1)
                ret_r.append(sr)
            ret.append(torch.stack(ret_r))
        return ret
    def compute_sent_average_loss(self, sent_output, target_sents):
        sentences_average_loss = None
        for i in range(len(sent_output)):
            sal = self.bce_loss(sent_output[i], target_sents[i].float())
            if(sentences_average_loss is None):
                sentences_average_loss  = sal / float(len(sent_output))
            else:
                sentences_average_loss += sal / float(len(sent_output))
        return sentences_average_loss
    def apply_masks_on_similarity(self, sentences, question, similarity):
        for bi in range(len(sentences)):
            qq = question[bi]
            qq = ( qq > 1).float()
            for si in range(len(sentences[bi])):
                ss  = sentences[bi][si]
                ss  = (ss > 1).float()
                sim_mask1 = qq.unsqueeze(-1).expand_as(similarity[bi][si])
                sim_mask2 = ss.unsqueeze(0).expand_as(similarity[bi][si])
                similarity[bi][si] *= sim_mask1
                similarity[bi][si] *= sim_mask2
        return similarity
    def forward(self, sentences, question, target_sents, target_docs, similarity_one_hot):
        #
        question                = [autograd.Variable(torch.LongTensor(item), requires_grad=False) for item in question]
        sentences               = [[autograd.Variable(torch.LongTensor(item), requires_grad=False) for item in item2] for item2 in sentences]
        #
        target_sents            = [autograd.Variable(torch.LongTensor(ts), requires_grad=False) for ts in target_sents]
        target_docs             = autograd.Variable(torch.LongTensor(target_docs), requires_grad=False)
        #
        question_embeds         = self.get_embeds(question)
        q_conv_res              = self.apply_convolution(question_embeds, self.quest_filters_conv)
        #
        sents_embeds            = [self.get_embeds(s) for s in sentences]
        s_conv_res              = [self.apply_convolution(s, self.sent_filters_conv) for s in sents_embeds]
        #
        similarity_insensitive  = [self.my_cosine_sim_many(question_embeds[i], sents_embeds[i]) for i in range(len(sents_embeds))]
        similarity_insensitive  = self.apply_masks_on_similarity(sentences, question, similarity_insensitive)
        similarity_sensitive    = [self.my_cosine_sim_many(q_conv_res[i], s_conv_res[i]) for i in range(len(q_conv_res))]
        similarity_one_hot      = [[autograd.Variable(torch.FloatTensor(item).transpose(0,1), requires_grad=False) for item in item2] for item2 in similarity_one_hot]
        #
        similarity_insensitive_pooled   = [[self.pooling_method(item) for item in item2] for item2 in similarity_insensitive]
        similarity_sensitive_pooled     = [[self.pooling_method(item) for item in item2] for item2 in similarity_sensitive]
        similarity_one_hot_pooled       = [[self.pooling_method(item) for item in item2] for item2 in similarity_one_hot]
        #
        sent_output             = self.get_sent_output(similarity_one_hot_pooled, similarity_insensitive_pooled, similarity_sensitive_pooled)
        sentences_average_loss  = self.compute_sent_average_loss(sent_output, target_sents)
        #
        document_emitions       = torch.stack([ s.max(-1)[0] for s in sent_output])
        document_average_loss   = self.bce_loss(document_emitions, target_docs.float())
        total_loss              = (sentences_average_loss + document_average_loss) / 2.0
        #
        return(total_loss, sent_output, document_emitions) # return the general loss, the sentences' relevance score and the documents' relevance scores

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

nof_cnn_filters = 12
filters_size    = 3
k_for_maxpool   = 5
lr              = 0.01
#
print('LOADING embedding_matrix (14GB)')
matrix          = np.load('/home/dpappas/joint_task_list_batches/embedding_matrix.npy')
t2i             = pickle.load(open('/home/dpappas/joint_task_list_batches/t2i.p','rb'))
print('Done')

model           = Posit_Drmm_Modeler(nof_filters=nof_cnn_filters, filters_size=filters_size, pretrained_embeds=matrix, k_for_maxpool=k_for_maxpool)
params          = list(set(model.parameters()) - set([model.word_embeddings.weight]))
print_params(model)
del(matrix)

resume_from         = '/home/dpappas/best_checkpoint.pth.tar'
load_model_from_checkpoint(resume_from)

abs_path            = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_docset_top100.test.pkl'
all_abs             = pickle.load(open(abs_path,'rb'))
bm25_scores_path    = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq_bm25_top100.test.pkl'
bm25_scores         = pickle.load(open(bm25_scores_path, 'rb'))

data = {}
data['questions'] = []
for quer in tqdm(bm25_scores['queries']):
    dato = {
        'body'      : quer['query_text'],
        'id'        : quer['query_id'],
        'documents' : []
    }
    doc_res = {}
    for retr in quer['retrieved_documents']:
        #
        doc_id      = retr['doc_id']
        #
        doc_title   = get_sents(all_abs[doc_id]['title'])
        doc_text    = get_sents(all_abs[doc_id]['abstractText'])
        all_sents   = doc_title + doc_text
        all_sents   = [s for s in all_sents if (len(bioclean(s)) > 0)]
        sents_inds  = [[get_index(token, t2i) for token in bioclean(s)] for s in all_sents]
        #
        quest_inds = [get_index(token, t2i) for token in bioclean(quer['query_text'])]
        #
        all_sims = [get_sim_mat(stoks, quest_inds) for stoks in sents_inds]
        #
        cost_, sent_ems, doc_ems = model(
            sentences=          [sents_inds],
            question=           [quest_inds],
            target_sents=       [[0] * len(sents_inds)],
            target_docs=        [0.0],
            similarity_one_hot= [all_sims]
        )
        #
        doc_res[doc_id] = float(doc_ems)
    doc_res             = sorted(doc_res.keys(), key=lambda x: doc_res[x], reverse=True)
    doc_res             = ["http://www.ncbi.nlm.nih.gov/pubmed/{}".format(pm) for pm in doc_res[:100]]
    dato['documents']   = doc_res
    data['questions'].append(dato)

with open('/home/dpappas/elk_relevant_abs_posit_drmm_lists.json', 'w') as f:
    f.write(json.dumps(data, indent=4, sort_keys=True))


# all_data    = pickle.load(open('joint_task_data_test.p','rb'))
# fpath       = '/home/DATA/Biomedical/document_ranking/bioasq_data/bioasq.test.json'
# data        = json.load(open(fpath, 'r'))
# total       = len(data['questions'])
# m           = 0
# for quest in data['questions']:
#     pprint(quest)
#     for item in [ d for d in all_data if(d['question'] == quest['body'])]:
#         #
#         sents_inds  = [[get_index(token, t2i) for token in bioclean(s)] for s in item['all_sents']]
#         quest_inds  = [get_index(token, t2i) for token in bioclean(item['question'])]
#         all_sims    = [get_sim_mat(stoks, quest_inds) for stoks in sents_inds]
#         sent_y      = np.array(item['sent_sim_vec'])
#         #
#         cost_, sent_ems, doc_ems = model(
#             sentences=          [sents_inds],
#             question=           [quest_inds],
#             target_sents=       [sent_y],
#             target_docs=        [item['doc_rel']],
#             similarity_one_hot= [all_sims]
#         )
#         print(item['doc_rel'], float(doc_ems))
#     break





