
__author__ = 'Dimitris'

import  json
import  torch
import  torch.nn.functional         as F
import  torch.nn                    as nn
import  numpy                       as np
import  torch.autograd              as autograd
from    tqdm                        import tqdm
from    gensim.models.keyedvectors  import KeyedVectors
import  pickle, re, os, collections, random
from    nltk.corpus import stopwords
import hnswlib
import torch.backends.cudnn as cudnn

my_seed     = 1
np.random.seed(my_seed)
random.seed(my_seed)
cudnn.benchmark = True
torch.manual_seed(my_seed)
print(torch.get_num_threads())
print(torch.cuda.is_available())
print(torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
if(use_cuda):
    torch.cuda.manual_seed(my_seed)

eng_stopwords   = stopwords.words('english')
nested_dict     = lambda: collections.defaultdict(nested_dict)
bioclean        = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

index_dir   = '/media/dpappas/dpappas_data/models_out/batched_semir_bioasq_2020/index/'
bin_fpaths  = [fpath for fpath in os.listdir(index_dir) if fpath.endswith('.bin')]

def get_vec(tok):
    try:
        return wv[tok]
    except:
        return unk_vec

def load_model_from_checkpoint(resume_from):
    global model
    if os.path.isfile(resume_from):
        print("=> loading checkpoint '{}'".format(resume_from))
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))

class Modeler(nn.Module):
    def __init__(self, emb_size = 30):
        super(Modeler, self).__init__()
        ###################################
        self.emb_size   = emb_size
        ###################################
        self.sent_attention = nn.Linear(in_features=50, out_features=2).to(device)
        self.gram_attention = nn.Linear(in_features=50, out_features=1).to(device)
        ###################################
        self.conv1 = nn.Conv1d(emb_size, 100, 3, padding=1)
        self.conv2 = nn.Conv1d(100, 50, 3, padding=1)
        ###################################
        self.cosine_sim = nn.CosineSimilarity(dim=2, eps=1e-6).to(device)
        ###################################
    def my_hinge_loss(self, positives, negatives, margin=1.0):
        delta       = negatives - positives
        loss_q_pos  = torch.sum(F.relu(margin + delta), dim=-1)
        return loss_q_pos
    def my_cosine_sim_3D(self, A, B):
        A_mag = torch.norm(A, 2, dim=2)
        B_mag = torch.norm(B, 2, dim=2)
        num = torch.bmm(A, B.transpose(-1, -2))
        den = torch.bmm(A_mag.unsqueeze(-1), B_mag.unsqueeze(-1).transpose(-1, -2))
        den = den + 1e-4  # added this cause it ended dividing with zero thus NAN!
        dist_mat = num / den
        return dist_mat
    def attention_similarity(self, input1):
        A           = input1
        B           = input1
        sim         = self.my_cosine_sim_3D(A, B)
        diagwnios   = (torch.eye(input1.size(1)).to(device) != 1).float().unsqueeze(0).expand_as(sim)
        return sim * diagwnios
    def masked_softmax(self, matrix, mask, dim=1, epsilon=1e-5):
        exps        = torch.exp(matrix)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps / masked_sums)
    def encode_sent(self, sent_vecs, sent_masks):
        sent_vecs       = autograd.Variable(torch.FloatTensor(sent_vecs), requires_grad=False).to(device)
        sent_masks      = autograd.Variable(torch.FloatTensor(sent_masks), requires_grad=False).to(device)
        sent_vecs       = torch.tanh(self.conv1(sent_vecs.transpose(-1, -2)))
        sent_vecs       = torch.tanh(self.conv2(sent_vecs).transpose(-1, -2))
        sent_att        = self.sent_attention(sent_vecs)
        sent_att        = self.masked_softmax(sent_att, sent_masks.unsqueeze(-1).expand_as(sent_att), dim=1, epsilon=1e-5)
        sent_vec        = torch.bmm(sent_att.transpose(-1, -2), sent_vecs)
        return sent_vec
    def forward(self, gram_vecs, sent_vecs, gram_masks, sent_masks, target):
        gram_vecs       = autograd.Variable(torch.FloatTensor(gram_vecs), requires_grad=False).to(device)
        sent_vecs       = autograd.Variable(torch.FloatTensor(sent_vecs), requires_grad=False).to(device)
        gram_masks      = autograd.Variable(torch.FloatTensor(gram_masks), requires_grad=False).to(device)
        sent_masks      = autograd.Variable(torch.FloatTensor(sent_masks), requires_grad=False).to(device)
        target          = autograd.Variable(torch.LongTensor(target), requires_grad=False).to(device)
        ################################
        gram_vecs       = self.conv1(gram_vecs.transpose(-1, -2))
        gram_vecs       = torch.tanh(gram_vecs)
        gram_vecs       = self.conv2(gram_vecs).transpose(-1, -2)
        gram_vecs       = torch.tanh(gram_vecs)
        ################################
        sent_vecs       = self.conv1(sent_vecs.transpose(-1, -2))
        sent_vecs       = torch.tanh(sent_vecs)
        sent_vecs       = self.conv2(sent_vecs).transpose(-1, -2)
        sent_vecs       = torch.tanh(sent_vecs)
        ################################
        gram_att        = self.gram_attention(gram_vecs)
        sent_att        = self.sent_attention(sent_vecs)
        gram_att        = self.masked_softmax(gram_att, gram_masks.unsqueeze(-1).expand_as(gram_att), dim=1, epsilon=1e-5)
        sent_att        = self.masked_softmax(sent_att, sent_masks.unsqueeze(-1).expand_as(sent_att), dim=1, epsilon=1e-5)
        ################################
        same_sim_neg    = self.attention_similarity(sent_att.transpose(0, 1))
        same_sim_neg    = torch.clamp(same_sim_neg, min=0., max=1.0)
        ################################
        gram_vec        = torch.bmm(gram_att.transpose(-1, -2), gram_vecs)
        sent_vec        = torch.bmm(sent_att.transpose(-1, -2), sent_vecs)
        ################################
        sent_sim        = self.cosine_sim(gram_vec.expand_as(sent_vec), sent_vec)
        ################################
        max_sent_sim, indices   = torch.max(sent_sim, dim=1)
        max_sent_sim            = torch.clamp(max_sent_sim, min=0., max=1.0)
        ################################
        loss1           = F.binary_cross_entropy(max_sent_sim, target.float())
        loss2           = same_sim_neg.sum() / same_sim_neg.numel()
        loss            = loss1 + 0.01 * loss2
        ################################
        return loss

##############################################################################################################

device          = torch.device("cuda") if(use_cuda) else torch.device("cpu")
model           = Modeler().to(device)

print('Loading pretrained W2V')
w2v_bin_path    = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
wv              = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
unk_vec         = np.average(wv.vectors, 0)

resume_from = '/media/dpappas/dpappas_data/models_out/batched_semir_bioasq_2020/checkpoint_27_0.272904634475708.pth.tar'
load_model_from_checkpoint(resume_from)

##############################################################################################################

def print_some_sents(max_dist_sent):
    global printed_one
    print(20 * '-')
    for lab, dist in zip(sent_labels1[0].tolist(), sent_distances1[0].tolist()):
        if(dist<=max_dist_sent):
            print('{} ~ {} ~ {}'.format(dist, lab, labels2names[lab]))
    print(20 * '-')
    for lab, dist in zip(sent_labels2[0].tolist(), sent_distances2[0].tolist()):
        if(dist<=max_dist_sent):
            print('{} ~ {} ~ {}'.format(dist, lab, labels2names[lab]))

def retrieve_some_sents():
    ret = []
    for lab, dist in zip(sent_labels1[0].tolist(), sent_distances1[0].tolist()):
        ret.append((dist, lab, labels2names[lab]))
    for lab, dist in zip(sent_labels2[0].tolist(), sent_distances2[0].tolist()):
        ret.append((dist, lab, labels2names[lab]))
    return sorted(ret, key=lambda x: x[0])

##############################################################################################################

dim                 = 50
space               = 'cosine'
max_elements        = 10000000

##############################################################################################################
bioasq_test_set_fpath   = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_1/BioASQ-task7bPhaseA-testset1'
qdata                   = json.load(open(bioasq_test_set_fpath))
questions               = [q['body'] for q in  qdata['questions']]
##############################################################################################################

# question = '''In CT imaging for the diagnosis or staging of acute diverticulitis what is the test accuracy of CT imaging for the diagnosis or staging of acute diverticulitis ?'''
# question = '''what is the test accuracy of CT imaging for the diagnosis or staging of acute diverticulitis ?'''

##############################################################################################################

max_dist = 0.05
results = {}
for bin_fpath in tqdm(bin_fpaths):
    # bin_fpath           = bin_fpaths[0]
    sent_index_path     = os.path.join(index_dir, bin_fpath)
    labels2names_path   = os.path.join(index_dir, 'labels2names{}.p'.format(bin_fpath.replace('my_index','').replace('.bin','')))
    ####################################################################################
    print("\nLoading labels2names'{}'\n".format(sent_index_path))
    labels2names        = pickle.load(open(labels2names_path, 'rb'))
    ####################################################################################
    sent_index          = hnswlib.Index(space, dim)
    print("\nLoading index from '{}'\n".format(sent_index_path))
    sent_index.load_index(sent_index_path, max_elements)
    sent_index.set_num_threads(4)
    ####################################################################################
    for question in questions:
        question_vecs = np.stack([get_vec(tok) for tok in bioclean(question)], 0)
        sent_mask = np.ones(question_vecs.shape[0])
        question_vecs = model.encode_sent([question_vecs], [sent_mask]).squeeze(0).cpu().data.numpy()
        ####################################################################################
        sent_labels1, sent_distances1 = sent_index.knn_query(question_vecs[0], k=25)
        sent_labels2, sent_distances2 = sent_index.knn_query(question_vecs[1], k=25)
        ####################################################################################
        # print_some_sents(max_dist_sent=max_dist)
        retrieved_sents = retrieve_some_sents()
        if(question not in results):
            results[question] = retrieved_sents
        else:
            results[question].extend(retrieved_sents)
        results[question] = sorted(results[question], key=lambda x: x[0])
    ####################################################################################

with open('bioasq_results.json', 'w') as of:
    of.write(json.dumps(results, indent=4, sort_keys=True))
    of.close()

import json

dd  = json.load(open('bioasq_results.json'))

dd3 = {}
for question, v in dd.items():
    if(question not in dd3):
        dd3[question] = {}
    for score, _, ss in v:
        pmid = ss.split(':')[0]
        try:
            dd3[question][pmid] += (1-score)
        except:
            dd3[question][pmid] = (1-score)
    dd3[question] = sorted(list(dd3[question].items()), key=lambda x:x[1], reverse=True)[:10]

with open('bioasq_results_scores.json', 'w') as of:
    of.write(json.dumps(dd3, indent=4, sort_keys=True))
    of.close()

