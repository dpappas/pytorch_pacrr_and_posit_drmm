

from emit_given_text import get_results_for_one_question
from pprint import pprint
from tqdm import tqdm
import pickle

key_questions = [
    [
        'In CT imaging for the diagnosis or staging of acute diverticulitis '
        'what is the test accuracy of CT imaging for the diagnosis or staging of acute diverticulitis?',
        #######################################################################################
        'In CT imaging for the diagnosis or staging of acute diverticulitis '
        'What are the effects of CT imaging on clinical outcomes and changes in clinical management?',
        #######################################################################################
        'In CT imaging for the diagnosis or staging of acute diverticulitis '
        'What are the downstream outcomes related to false positive or false negative CT readings of '
        'acute uncomplicated or complicated diverticulitis',
        #######################################################################################
        'In CT imaging for the diagnosis or staging of acute diverticulitis '
        'For patients presenting with acute abdominal pain, with the possibility of acute diverticulitis, '
        'what are the downstream outcomes related to incidental findings (e.g., liver mass)'
    ],
    [
        'What are the benefits and harms of various treatment options for the treatment of acute diverticulitis?',
        'For patients with acute uncomplicated diverticulitis, what are the effectiveness and harms of hospitalization versus outpatient management of the acute episode?',
        'For patients with acute uncomplicated or complicated diverticulitis, what are the effects, comparative effects, and harms of antibiotics?',
        'For patients with acute complicated diverticulitis, what are the effects and harms of interventional radiology procedures compared with conservative management?',
    ],
    [
        'What are the benefits and harms of colonoscopy (or other colon imaging test) following an episode of acute diverticulitis?',
        'What is the incidence of malignant and premalignant colon tumors found by colonoscopy, and what is the incidence of colon cancer mortality among patients undergoing screening?',
        'What are the procedure-related and other harms of colonoscopy or CT colonography?',
        'What is the frequency of inadequate imaging due to intolerance or technical feasibility?'
    ],
    [
        'What are the effects, comparative effects, and harms of pharmacological interventions (e.g., mesalamine), non-pharmacological interventions (e.g., medical nutrition therapy), and elective surgery to prevent recurrent diverticulitis?'
    ]
]

results = []
for kqs in tqdm(key_questions):
    res = []
    for kq in kqs:
        ret_dummy       = get_results_for_one_question(kq, how_many=100)
        res.append(ret_dummy)
    results.append(res)

pickle.dump(results, open('results_SR.p', 'wb'))

'''
import pickle
from pprint import pprint
from collections import Counter
results = pickle.load(open('results_SR.p', 'rb'))
# pprint(results[0])

documents = []
all_q_res = []
for key_question_results in results:
    docs = []
    for question_res in key_question_results:
        for k2, v2 in question_res.items():
            if(v2['doc_score'] > 0.3):
                print(k2, v2['doc_score'])
                docs.append(k2)
                all_q_res.append(k2)
    documents.append(Counter(docs))

all_q_res = Counter(all_q_res)
print(len(documents))
pprint(documents[0])
pprint(all_q_res)
pprint(len(all_q_res))


'''



