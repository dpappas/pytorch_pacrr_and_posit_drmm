
import torch, re, random, spacy, time, pickle
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch import FloatTensor as FT
from my_data_handling import DataHandler
from pprint import pprint

my_seed = 1989
random.seed(my_seed)
torch.manual_seed(my_seed)

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

######################################################################################################
use_cuda    = torch.cuda.is_available()
device      = torch.device("cuda") if(use_cuda) else torch.device("cpu")
######################################################################################################
en          = spacy.load('en_core_web_sm')

data_path       = '/home/dpappas/quora_duplicate_questions.tsv'
data_handler    = DataHandler(data_path)
data_handler.load_model('datahandler_model.p')



