
# pip install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz

import scispacy
from scispacy.abbreviation import AbbreviationDetector
import spacy
import en_core_sci_lg
from spacy.matcher import Matcher
from pprint import pprint
from my_sentence_splitting import get_sents
import json
from tqdm import tqdm
import re

nlp = spacy.load("en_core_sci_lg")

abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)

matcher = Matcher(nlp.vocab)

age_words     = [
    "boy", "girl", "man", "woman", 'men', 'women', 'girls', 'boys', 'baby', 'babies', 'infant',
    'male', 'female', 'males', 'females', 'adult', 'adults', 'children', 'child', 'newborn', 'neonates',
    'toddlers', 'neonate', 'toddler', 'adolescent', 'adolescents', 'elderly', 'young'
]

age_pattern_1 = [  # this is simple. It will match everything
    {"OP": "+", "LOWER": {"IN": age_words}, "POS": {"IN": ["NOUN"]}},
    {"OP": "?", "LOWER": {"IN": ['patient', 'patients']}}
]
age_pattern_2 = [  # this is simple. It will match everything
    {"OP": "*", "LOWER": {"IN":["over", "under"]}},
    {"POS": {"IN":["NUM"]}},
    {"LOWER": {"IN":["-", "±"]}},
    {"POS": {"IN":["NUM"]}},
    {"LOWER": "years"},
    {"LOWER": "old"},
]
age_pattern_3 = [  # this is simple. It will match everything
    {"OP": "*", "LOWER": {"IN":["over", "under"]}},
    {"POS": {"IN":["NUM"]}},
    {"LOWER": "years"},
    {"LOWER": "old"},
]
age_pattern_4 = [  # this is simple. It will match everything
    {"LOWER": "up"},
    {"LOWER": "to"},
    {"POS": {"IN":["NUM"]}},
    {"LOWER": "years"},
    {"LOWER": "old"},
]
age_pattern_5 = [  # this is simple. It will match everything
    {"LOWER": "from"},
    {"POS": {"IN":["NUM"]}},
    {"OP": "*", "LOWER": "years"},
    {"OP": "*", "LOWER": "old"},
    {"LOWER": "to"},
    {"POS": {"IN":["NUM"]}},
    {"LOWER": "years"},
    {"LOWER": "old"},
]
age_pattern_6 = [  # this is simple. It will match everything
    {"POS": {"IN":["NUM"]}},
    {"OP": "*", "LOWER": "-"},
    {"LOWER": {"IN":["year", 'years']}},
    {"OP": "*", "LOWER": "-"},
    {"LOWER": "old"},
    {"LOWER": {"IN":age_words}, "POS": {"IN":["NOUN"]}},
]
age_pattern_7 = [  # this is simple. It will match everything
    {"LOWER": {"IN": age_words}, "POS": {"IN":["NOUN"]}},
    {"LOWER": {"IN":["between"]}},
    {"POS": {"IN": ["NUM"]}},
    {"LOWER": {"IN": ["and"]}},
    {"POS": {"IN":["NUM"]}},
    {"LOWER": {"IN": ["year", "years"]}},
    {"OP": "*", "LOWER": "old"}
]
age_pattern_8 = [  # this is simple. It will match everything
    {"LOWER": {"IN":["less", "more"]}},
    {"LOWER": {"IN": ["than"]}},
    {"POS": {"IN":["NUM"]}},
    {"LOWER": "years"},
    {"LOWER": "old"},
]
age_pattern_9 = [
    {"LOWER": {"IN": ["age", "ages"]}},
    {"POS": {"IN": ["NUM"]}},
    {"LOWER": {"IN": ["to"]}},
    {"POS": {"IN": ["NUM"]}},
    {"LOWER":{"IN": ["years", 'year']}},
    {"OP": "?", "LOWER": "old"},
]
age_pattern_10 = [
    {"LOWER": {"IN": ["patient", "patients", 'probands', 'proband']}, "POS": {"IN":["NOUN"]}},
    {"LOWER": {"IN": ["aged"]}},
    {"POS": {"IN": ["NUM"]}},
    {"OP": "?", "LOWER": {"IN": ["month", 'months', 'year', 'years']}},
    {"LOWER": {"IN": ["to"]}},
    {"POS": {"IN": ["NUM"]}},
    {"LOWER":{"IN": ["years", 'year', 'months']}}
]
# ages 6 to 25 years
# patients aged 6 months to 18 years

matcher.add("AGE", None, age_pattern_1)
matcher.add("AGE", None, age_pattern_2)
matcher.add("AGE", None, age_pattern_3)
matcher.add("AGE", None, age_pattern_4)
matcher.add("AGE", None, age_pattern_5)
matcher.add("AGE", None, age_pattern_6)
matcher.add("AGE", None, age_pattern_7)
matcher.add("AGE", None, age_pattern_8)
matcher.add("AGE", None, age_pattern_9)
matcher.add("AGE", None, age_pattern_10)

patient_pattern_1 = [  # this is simple. It will match everything
    {"POS": {"IN":["NUM"]}},
    {"OP": "+", "POS": {"IN":["ADJ", "PROPN", "NOUN", "CCONJ", "NUM"]}},
    {"LOWER": {"IN":['patient', 'patients']}},
]
patient_pattern_2 = [  # this is simple. It will match everything
    {"OP": "+", "POS": {"IN":["NUM"]}},
    {"LOWER": {"IN": ['pregnant']}},
    {"LOWER": {"IN": ['woman', 'women']}},
]
patient_pattern_3 = [  # this is simple. It will match everything
    {"POS"  : {"IN":["NUM"]}},
    {"LOWER": {"IN":['infant', 'infants']}},
]

matcher.add("PATIENTS NO", None, patient_pattern_1)
matcher.add("PATIENTS NO", None, patient_pattern_2)
matcher.add("PATIENTS NO", None, patient_pattern_3)

cond_patt_1 = [  # this is simple. It will match everything
    {"LOWER": {"IN": ['symptom', 'symptoms']}},
    {"LOWER": {"IN": ['of']}},
    {"OP": "+", "POS": {"IN":["ADJ", "PROPN", "NOUN", "CCONJ", "NUM"]}},
    {"OP": "*", "LOWER": {"IN":["("]}},
    {"OP": "*", "POS": {"IN":["ADJ", "PROPN", "NOUN", "CCONJ", "NUM"]}},
    {"OP": "*", "LOWER": {"IN": [")"]}},
    {"OP": "*", "POS": {"IN":["ADJ", "PROPN", "NOUN", "CCONJ", "NUM"]}},
]
matcher.add("CONDITION", None, cond_patt_1)

cond_patt_2 = [  # this is simple. It will match everything
    {"OP": "+", "LOWER": {"IN": age_words+['patient', 'patients']}},
    {"LOWER": {"IN": ['with']}},
    {"OP": "?", "LOWER": {"IN": ['confirmed']}},
    {"OP": "+", "POS": {"IN":["ADJ", "PROPN", "NOUN", "CCONJ", "NUM", "ADV"]}},
    {"OP": "*", "LOWER": {"IN":["("]}},
    {"OP": "*", "POS": {"IN":["ADJ", "PROPN", "NOUN", "CCONJ", "NUM", "ADV"]}},
    {"OP": "*", "LOWER": {"IN": [")"]}},
    {"OP": "*", "POS": {"IN":["ADJ", "PROPN", "NOUN", "CCONJ", "NUM", "ADV"]}},
]

matcher.add("CONDITION", None, cond_patt_2)

cond_patt_3 = [  # this is simple. It will match everything
    {"OP": "+", "POS": {"IN":["ADJ", "PROPN", "NOUN", "CCONJ", "NUM"]}},
    {"OP": "+", "LOWER": {"IN": age_words+['patient', 'patients']}}
]

matcher.add("CONDITION", None, cond_patt_3)

# pattern2 = [
#     # {"POS": "ADJ", "OP": "*"},
#     # {"POS": "NOUN", "OP": "+"},
#     {"POS": {'NOT_IN':["NOUN"]}, "OP": "*"},
#     {"LEMMA": {"IN": [
#         # "increase",
#         # 'raise',
#         # 'increment',
#         # 'rise',
#         'upturn', 'boost', 'downregulate'
#     ]}},
#     {"POS": {'NOT_IN':["NOUN"]}, "OP": "*"},
#     {"POS": "NOUN", "OP": "+"}
# ]
# matcher.add("DOWN", None, pattern2)
#

def get_entities_and_abbrev(text):
    doc = nlp(text)
    if(len(doc.ents)):
        pprint(dict((ent.text, (ent.label_, ent.start, ent.end)) for ent in doc.ents))
    if(len(doc._.abbreviations)):
        print(doc._.abbreviations)

# doc = nlp('Twenty-nine eligible patients')
# print([(token.text, token.lemma_, token.pos_) for token in doc])
# exit()

def do_for_sent(sent, printout=False):
    doc = nlp(sent)
    if (printout):
        print([(token.text, token.lemma_, token.pos_) for token in doc])
    matches = matcher(doc)
    all_ncs = set([nc.text.lower() for nc in doc.noun_chunks])
    for tok in doc:
        if tok.pos_ in ['NOUN', 'PROPN']:
            if not any(tok.text.lower() in nc for nc in all_ncs):
                all_ncs.add(tok.text.lower())
    kept_phrases = []
    for match_id, start, end in matches:
        string_id   = nlp.vocab.strings[match_id]  # Get string representation
        span        = doc[start:end]  # The matched span
        kept_phrases.append(span.text)
        # if(printout):
        #     print(string_id, span.text)
    kept_phrases = set(kept_phrases)
    kept_phrases_2 = []
    kp_concat = ' || '.join(kept_phrases)
    for phrase in kept_phrases:
        if(kp_concat.count(phrase) ==1):
            kept_phrases_2.append(phrase)
    return kept_phrases_2
    # return set(kept_phrases)

def do_for_sent_1(sent):
    doc = nlp(sent)
    matches = matcher(doc)
    flag = False
    kept_phrases = []
    for match_id, start, end in matches:
        span = doc[start:end]  # The matched span
        #################################################
        if(
            span[0].pos_ in ['NOUN', "ADJ"] and
            span[-1].pos_ in ['NOUN'] and
            sum(int(token.pos_ in ['NOUN']) for token in span) <= 10 and
            len(span) <= 12
        ):
            # print(span.text)
            kept_phrases.append(span.text)
            flag = True
    if(flag):
        kept_phrases_2 = []
        kp_concat = ' || '.join(kept_phrases)
        for phrase in kept_phrases:
            if(kp_concat.count(phrase) ==1):
                kept_phrases_2.append(phrase)
        print('\n'.join(kept_phrases_2))
        # print([(token.text, token.lemma_, token.pos_) for token in doc])
    # print(sent)

def do_for_doc(all_text):
    segments    = [all_text.split('------------------------------')[0].strip()]
    segments    = segments + [t.strip() for t in all_text.split('------------------------------')[1].strip().split('\n')]
    all_phrases = []
    for segment in segments:
        for sent in get_sents(segment):
            sent = re.sub(r'\([^()]*\)', '', sent)
            sent = re.sub(r'__.*?__', '', sent)
            sent = re.sub(r'\s+', ' ', sent)
            if(len(sent)>10):
                kept_phrases = do_for_sent(sent)
                all_phrases.extend(kept_phrases)
                # do_for_sent_1(sent)
    return all_phrases

if __name__ == '__main__':
    texts = [
        # "Five of the seven patients presented with symptoms of COVID-19, including cough, myalgias, fevers, chest pain, and headache.",
        # 'It can be difficult for adolescents with inflammatory bowel disease (IBD) to make the transition from paediatric to adult care.',
        # 'Association between miR-200c and the survival of patients with stage I epithelial ovarian cancer: a retrospective study of two independent tumour tissue collections.',
        # 'To compare the efficiency of releasable scleral buckling (RSB) and pars plana vitrectomy (PPV) in the treatment of phakic patients with primary rhegmatogenous retinal detachment.',
        # 'To evaluate the changes in activity of biomarkers of Mu[Combining Diaeresis]ller cells (MC) in aqueous humor of patients with diabetic macular edema after subthreshold micropulse laser, over 1 year.',
        # 'Rates of Adverse IBD-Related Outcomes for Patients With IBD and Concomitant Prostate Cancer Treated With Radiation Therapy.',
        # 'External and middle ear resonance frequency of fourty patients with tympanoplasty and mastoidectomy.',
        # 'To describe the late results of the placement of skin graft over conjunctiva-Müller muscle complex in 3 patients with ablepharon-macrostomia syndrome (AMS) and to review the procedures used to manage the upper eyelids in AMS.',
        # 'There are unique considerations for many adult patients with gliomas who are vulnerable to the novel coronavirus due to older age and immunosuppression.',
        # 'As patients with terminal illnesses, they present ethical challenges for centers that may need to ration access to ventilator care due to insufficient critical care capacity.',
        # 'This study focused on determining risks from stereotactic radiotherapy using flattening filter-free (FFF) beams for patients with cardiac implantable electronic device (CIEDs).',
        # 'This research aims to clarify to what extent dengue patients were managed in accordance with the guidelines in Japan.',
        # 'Pre-death grief in caregivers of Alzheimer patients.',
        'This study examined memory and executive functions of switching and distributing attention in 25 Alzheimer patients (AD), 9 patients with frontal variant of frontotemporal dementia (fvFTD), and 25 healthy older people, as a control group, in three tasks: verbal digit span, Brown-Peterson (B-P) task, and dual-task.',
        'Similarly, by now, there have been over 60 cases of pregnant women with confirmed COVID-19 in China and the vast majority of these women have had mild to moderate pneumonia.',
        "In a previous study of MERS-CoV, there were 11 pregnant women [7] .",
        "Of 9 pregnant women, 4 (44%) had premature delivery [5] .",
    ]
    for text in texts:
        pprint(do_for_sent(text, printout=True))


