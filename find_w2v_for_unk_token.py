from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import re
import numpy as np

softmax  	= lambda z: np.exp(z) / np.sum(np.exp(z))
bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

model       = KeyedVectors.load_word2vec_format(datapath("/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin"), binary=True)

def get_for_window(tokens_in_window):
    h       = np.array([model.wv.word_vec(t) for t in tokens_in_window])
    h       = np.mean(h, axis=0)
    z       = np.matmul(model.wv.syn0, h)
    y       = softmax(z)
    vec     = np.matmul( y.reshape((1,-1)), model.wv.syn0).squeeze()
    ret     = model.similar_by_vector(vec, 5)
    return ret

# sentence    = 'The demand for inpatient and ICU beds for COVID-19 in the US: lessons from Chinese cities'

sentence    = '''
The demand for inpatient and ICU beds for COVID-19 in the US: lessons from Chinese cities
We summed the total patient-days under critical and/or severe condition to estimate the total ICU-days and serious-inpatient-days.
We plotted the raw number of patients in critical and severe conditions and patients hospitalized on each day for Wuhan and Guangzhou, and estimated the proportion of hospitalization and ICU admission per 10,000 adults based on the assumption that there were 9 million people present in Wuhan during the lockdown, 3 of whom 88.16% were age 15 or above (2010 census for cities in Hubei province), and 14.9 million present in Guangzhou of whom 82.82% were age 15 or above (Guangdong statistical bureau).
We then projected the number of patients who have severe and critical COVID-19 disease at the peak of a Wuhan-like outbreak in the 30 most populous US cities by assuming that the effect of age and comorbidity on patient outcomes would be the same as their effect on COVID-19 mortality as derived from case reports from China until February 11.
8 Specifically, we .
CC-BY 4.0 International license It is made available under a author/funder, who has granted medRxiv a license to display the preprint in perpetuity.
'''
tokens      = bioclean(sentence)

for i in range(len(tokens)):
    tok = tokens[i]
    try:
        vec = model[tok]
    except KeyError:
        print(tok)
        print(get_for_window(tokens[i-3:i]+tokens[i+1:i+3]))













