
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

age_words = [
    "boy", "girl", "man", "woman", 'men', 'women', 'girls', 'boys', 'baby', 'babies', 'infant',
    'male', 'female', 'males', 'females', 'adult', 'adults', 'children', 'child', 'newborn',
]

age_pattern_1 = [  # this is simple. It will match everything
    {"LOWER": {"IN":["young", 'elderly', 'newborn']}},
    {"LOWER": {"IN":age_words}}
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
    {"LOWER": {"IN":age_words}},
]
age_pattern_7 = [  # this is simple. It will match everything
    {"LOWER": {"IN": age_words}},
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

matcher.add("AGE", None, age_pattern_1)
matcher.add("AGE", None, age_pattern_2)
matcher.add("AGE", None, age_pattern_3)
matcher.add("AGE", None, age_pattern_4)
matcher.add("AGE", None, age_pattern_5)
matcher.add("AGE", None, age_pattern_6)
matcher.add("AGE", None, age_pattern_7)
matcher.add("AGE", None, age_pattern_8)

patient_pattern_1 = [  # this is simple. It will match everything
    {"POS": {"IN":["NUM"]}},
    {"OP": "?"},
    {"OP": "?"},
    {"LOWER": {"IN":['patient', 'patients']}},
]
patient_pattern_2 = [  # this is simple. It will match everything
    {"OP": "?", "POS": {"IN":["NUM"]}},
    {"LOWER": {"IN":['pregnant']}},
    {"LOWER": {"IN": ['woman', 'women']}},
]
patient_pattern_3 = [  # this is simple. It will match everything
    {"POS"  : {"IN":["NUM"]}},
    {"LOWER": {"IN":['infant', 'infants']}},
]

matcher.add("PATIENTS NO", None, patient_pattern_1)
matcher.add("PATIENTS NO", None, patient_pattern_2)
matcher.add("PATIENTS NO", None, patient_pattern_3)

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

def do_for_sent(sent):
    doc = nlp(sent)
    # pprint(ent._.umls_ents for ent in doc.ents)
    # exit()
    # print(50 * '-')
    matches = matcher(doc)
    flag = False
    all_ncs = set([nc.text.lower() for nc in doc.noun_chunks])
    for tok in doc:
        if tok.pos_ in ['NOUN', 'PROPN']:
            if not any(tok.text.lower() in nc for nc in all_ncs):
                all_ncs.add(tok.text.lower())
    kept_phrases = []
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        #################################################
        # print(match_id, string_id, start, end, span.text)
        # #################################################
        # pprint([(token.text, token.lemma_, token.pos_) for token in span])
        # pprint(list(span.noun_chunks))
        if(
            any(span.text.lower().startswith(nc) for nc in all_ncs)     # starts with noun chunk
            and
            any(span.text.lower().endswith(nc) for nc in all_ncs)       # ends with noun chunk
            and
            sum(int(nc in span.text.lower()) for nc in all_ncs) < 6     # has up to k noun chunks
            and
            len(span) <= 12                                             # has length up to k
        ):
            # print(string_id, span.text)
            kept_phrases.append(span.text)
            flag = True
            # print(sent)
            # print(all_ncs)
    if(flag):
        kept_phrases_2 = []
        kp_concat = ' || '.join(kept_phrases)
        for phrase in kept_phrases:
            if(kp_concat.count(phrase) ==1):
                kept_phrases_2.append(phrase)
        print('\n'.join(kept_phrases_2))
        # print(all_ncs)
        # print([(token.text, token.lemma_, token.pos_) for token in doc])
    # print(sent)

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

def do_for_text(all_text):
    segments    = [all_text.split('------------------------------')[0].strip()]
    segments    = segments + [t.strip() for t in all_text.split('------------------------------')[1].strip().split('\n')]
    for segment in segments:
        for sent in get_sents(segment):
            sent = re.sub(r'\([^()]*\)', '', sent)
            sent = re.sub(r'__.*?__', '', sent)
            sent = re.sub(r'\s+', ' ', sent)
            if(len(sent)>10):
                do_for_sent(sent)
                # do_for_sent_1(sent)

