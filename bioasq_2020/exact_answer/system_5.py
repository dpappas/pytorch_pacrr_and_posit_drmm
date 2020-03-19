
# python3.6
'''
get GOLD results and then call SGRANK to get the answers
GOLD -> SGRANK
'''

# import  scispacy
import  spacy
import  re
import  json, pickle
from tqdm import tqdm
from    textacy import make_spacy_doc, keyterms

bioclean_mod    = lambda t: re.sub('[~`@#$-=<>/.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').replace("\n", ' ').strip().lower())

nlp 	= spacy.load("en_core_sci_lg")

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
ofpath          = '/home/dpappas/system_5_gold_sgrank.json'
idf_pickle_path = '/home/dpappas/bioasq_all/idf.pkl'

idf, max_idf    = load_idfs(idf_pickle_path)

qid2type = dict((quest['id'], quest['type']) for quest in data['questions'])

for quest in tqdm(data['questions']):
    if(quest['type'] not in ['factoid', 'list']):
        continue
    # print(quest['body'])
    # print(20*'-')
    # my_ents     = []
    # my_kt       = []
    # for snip in quest['snippets']:
    #     my_kt.extend(
    #         [
    #             ph[0].lower() for ph in get_keyphrases_sgrank(snip['text'], idfs=idf)
    #             if(check_answer(ph[0], quest['body']))
    #         ]
    #     )
    #     my_ents.extend(
    #         [
    #             ph.lower() for ph in get_phrases(snip['text'])
    #             if(check_answer(ph, quest['body']))
    #         ]
    #     )
    ########################################
    my_kt_ens   = []
    all_snips = ' \n'.join(snip['text'] for snip in quest['snippets'])
    my_kt_ens.extend(
        [
            ph for ph in get_keyphrases_sgrank(all_snips, idfs=idf)
            if(check_answer(ph[0], quest['body']))
        ]
    )
    ########################################
    # pprint(Counter(my_kt).most_common(5))
    # # pprint(Counter(my_ents).most_common(5))
    # pprint(sorted(my_kt_ens, key= lambda x: x[1], reverse= True))
    # print(20*'=')
    ########################################
    if (quest['type'] == 'list'):
        quest['exact_answer'] = [[kt[0]] for kt in my_kt_ens]
    else:
        quest['exact_answer'] = [kt[0] for kt in my_kt_ens]

with open(ofpath, 'w') as of:
    of.write(json.dumps(data, indent=4, sort_keys=True))
    of.close()
