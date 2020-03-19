
# python3.6
'''
get GOLD results and then call SGRANK to get the answers
GOLD -> SGRANK
'''

import scispacy
import spacy
import re
import json, pickle
from pprint import pprint
from collections import Counter
from textacy import make_spacy_doc, keyterms

bioclean_mod    = lambda t: re.sub('[~`@#$-=<>/.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').replace("\n", ' ').strip().lower())

def load_idfs(idf_path):
    print('Loading IDF tables')
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    print('Loaded idf tables with max idf {}'.format(max_idf))
    return idf, max_idf

def get_phrases(text):
    phrases = []
    flag    = False
    doc = nlp(text)
    for tok in doc:
        tok_idf = idf[str(tok).lower()] if str(tok).lower() in idf else max_idf
        # print(str(tok), tok.pos_, tok_idf)
        if (tok.pos_ in ['NOUN', 'ADJ']):
            # if(tok_idf <= 5.0):
            if (flag):
                phrases[-1] += ' ' + tok.text
            else:
                phrases.append(tok.text)
            flag = False
            # else:
            #     flag = True
        else:
            flag = False
        # print(tok, tok.pos_)
    return phrases

def get_keyphrases_sgrank(text, idfs):
    doc = make_spacy_doc(bioclean_mod(text), lang='en')
    keyphrases = keyterms.sgrank(
        doc,
        ngrams       = tuple(range(1, 4)),
        normalize    = None,  # None, # u'lemma', # u'lower'
        window_width = 50,
        n_keyterms   = 5,
        idf          = None,
        include_pos  = ("NOUN", "PROPN", "ADJ"),  # ("NOUN", "PROPN", "ADJ"), # ("NOUN", "PROPN", "ADJ", "VERB", "CCONJ"),
    )
    if(len(keyphrases)==0):
        # print([(tok, idfs[tok] if tok in idfs else max_idf) for tok in doc if tok.pos=='NOUN'])
        toks_with_idfs  = [(tok, idfs[tok] if tok in idfs else max_idf) for tok in doc]
        toks_with_idfs  = sorted(toks_with_idfs, key=lambda x: x[1])
        keyphrases      = [(tt[0].text, tt[1]) for tt in toks_with_idfs]
    # return text, keyphrases
    return keyphrases

def check_answer(ans, quest):
    quest_toks  = bioclean_mod(quest).split()
    ans_toks    = bioclean_mod(ans).split()
    #############################################
    quest_toks  = [t[:-1] if t.endswith('s') else t for t in quest_toks]
    ans_toks    = [t[:-1] if t.endswith('s') else t for t in ans_toks]
    #############################################
    if(bioclean_mod(ans) in bioclean_mod(quest)):
        return False
    if(all(t in quest_toks for t in ans_toks)):
        return False
    return True

data            = json.load(open('/home/dpappas/BioASQ-task8bPhaseB-testset2'))
fpath3          = '/home/dpappas/bioasq8_batch2_system1_results.json'
f3              = '/home/dpappas/system_1_jpdrmm_sgrank.json'
idf_pickle_path = '/home/dpappas/bioasq_all/idf.pkl'
