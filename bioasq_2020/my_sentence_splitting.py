#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Dimitris'

import re
from pprint import pprint
from nltk.tokenize import sent_tokenize

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
    ind = [item.end() for item in re.finditer('[\W\s]sp.|[\W\s]nos.|[\W\s]figs.|[\W\s]sp.[\W\s]no.|[\W\s][vols.|[\W\s]cv.|[\W\s]fig.|[\W\s]e.g.|[\W\s]et[\W\s]al.|[\W\s]i.e.|[\W\s]p.p.m.|[\W\s]cf.|[\W\s]n.a.|[\W\s]min.', sent)]
    if(len(ind)==0):
        return False
    else:
        ind = max(ind)
        if (len(sent) == ind):
            return True
        else:
            return False

def starts_with_special(sent):
    sent    = sent.strip().lower()
    chars   = ':%@#$^&*()\\,<>?/=+-_'
    for c in chars:
        if(sent.startswith(c)):
            return True
    return False

def split_sentences2(text):
    sents = [l.strip() for l in sent_tokenize(text)]
    ret = []
    i = 0
    while (i < len(sents)):
        sent = sents[i]
        while (
            ((i + 1) < len(sents)) and
            (
                ends_with_special(sent) or
                not first_alpha_is_upper(sents[i+1]) or
                starts_with_special(sents[i + 1])
                # sent[-5:].count('.') > 1       or
                # sents[i+1][:10].count('.')>1   or
                # len(sent.split()) < 2          or
                # len(sents[i+1].split()) < 2
            )
        ):
            sent += ' ' + sents[i + 1]
            i += 1
        ret.append(sent.replace('\n', ' ').strip())
        i += 1
    return ret

def get_sents(ntext):
    sents = []
    for subtext in ntext.split('\n'):
        subtext = re.sub('\s+', ' ', subtext.replace('\n',' ')).strip()
        if (len(subtext) > 0):
            ss = split_sentences2(subtext)
            sents.extend([ s for s in ss if(len(s.strip())>0)])
    if(len(sents)>0 and len(sents[-1]) == 0 ):
        sents = sents[:-1]
    return sents

if __name__ == '__main__':
    example_text = '''
    Early postnatal dexamethasone therapy for the prevention of chronic lung disease in preterm infants with respiratory distress syndrome: a multicenter clinical trial.
    OBJECTIVES:
    To study whether early postnatal (<12 hours) dexamethasone therapy reduces the incidence of chronic lung disease in preterm infants with respiratory distress syndrome.
    MATERIALS AND METHODS:
    A multicenter randomized, double-blind clinical trial was undertaken on 262 (saline placebo, 130; dexamethasone, 132) preterm infants (<2000 g) who had respiratory distress syndrome and required mechanical ventilation shortly after birth. The sample size was calculated based on the 50% reduction in the incidence of chronic lung disease when early dexamethasone is used, allowing a 5% chance of a type I error and a 10% chance of a type II error. For infants who received dexamethasone, the dosing schedules were: 0.25 mg/kg/dose every 12 hours intravenously on days 1 through 7; 0.12 mg/kg/dose every 12 hours intravenously on days 8 through 14; 0.05 mg/kg/dose every 12 hours intravenously on days 15 through 21; and 0. 02 mg/kg/dose every 12 hours intravenously on days 22 through 28. A standard protocol for respiratory care was followed by the participating hospitals. The protocol emphasized the criteria of initiation and weaning from mechanical ventilation. The diagnosis of chronic lung disease based on oxygen dependence and abnormal chest roentgenogram was made at 28 days of age. To assess the effect of dexamethasone on pulmonary inflammatory response, serial tracheal aspirates were assayed for cell counts, protein, leukotriene B4, and 6-keto prostaglandin F1alpha. All infants were observed for possible side effects, including hypertension, hyperglycemia, sepsis, intraventricular hemorrhage, retinopathy of prematurity, cardiomyopathy, and alterations in calcium homeostasis, protein metabolism, and somatic growth.
    RESULTS:
    Infants in the dexamethasone group had a significantly lower incidence of chronic lung disease than infants in the placebo group either judged at 28 postnatal days (21/132 vs 40/130) or at 36 postconceptional weeks (20/132 vs 37/130). More infants in the dexamethasone group than in the placebo group were extubated during the study. There was no difference between the groups in mortality (39/130 vs 44/132); however, a higher proportion of infants in the dexamethasone group died in the late study period, probably attributable to infection or sepsis. There was no difference between the groups in duration of oxygen therapy and hospitalization. Early postnatal use of dexamethasone was associated with a significant decrease in tracheal aspirate cell counts, protein, leukotriene B4, and 6-keto prostaglandin F1alpha, suggesting a suppression of pulmonary inflammatory response. Significantly more infants in the dexamethasone group than in the placebo group had either bacteremia or clinical sepsis (43/132 vs 27/130). Other immediate, but transient, side effects observed in the dexamethasone group are: an increase in blood glucose and blood pressure, cardiac hypertrophy, hyperparathyroidism, and a transient delay in the rate of growth.
    CONCLUSIONS:
    In preterm infants with severe respiratory distress syndrome requiring assisted ventilation shortly after birth, early postnatal dexamethasone therapy reduces the incidence of chronic lung disease, probably on the basis of decreasing the pulmonary inflammatory process during the early neonatal period. Infection or sepsis is the major side effect that may affect the immediate outcome. Other observable side effects are transient. In view of the significant side effects and the lack of overall improvement in outcome and mortality, and the lack of long term follow-up data, the routine use of early dexamethasone therapy is not yet recommended.
    '''
    print('NLTK')
    for sent in sent_tokenize(example_text.strip()):
        print(sent)
    print('OUR')
    for sent in get_sents(example_text.strip()):
        print(sent)

