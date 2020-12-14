
import  json, re
from    collections import Counter
from    pprint import pprint
from    retrieve_and_rerank import retrieve_given_question
from    emit_exact_answers import emit_exact_answers
from    tqdm import tqdm

flattened = lambda l: [item for sublist in l for item in sublist]

# fpath   = '/home/dpappas/BioASQ-taskSynergy-dryRun-testset'
fpath   = '/home/dpappas/COVID_SYNERGY/BioASQ-taskSynergy-testset1'
d       = json.load(open(fpath))

nonos           = [
'sars',
'sars - cov',
'cov - 2',
'coronavirus',
'covid - 19',
'covid',
]
nonos_2         = ['et al', 'et. al', '>']
sent_min_chars  = 20

def get_answers(qtext, le_text):
    overall_exact   = emit_exact_answers(qtext, le_text) #
    overall_exact   = [
        t
        for t in overall_exact
        if (
            t[1] >= 0.5 and
            t[2] >= 0.5 and
            all(nono != t[0] for nono in nonos) and
            all(nono2 not in t[0] for nono2 in nonos_2)
        )
    ]
    overall_exact   = [t[0] for t in overall_exact]
    return overall_exact

for question in tqdm(d['questions']):
    qtype           = question['type']
    if(qtype != 'factoid' and qtype != 'list'):
        continue
    qtext                   = question['body']
    res                     = retrieve_given_question(qtext)
    ###############################################################################################################
    par_ex_ans_counter      = Counter()
    all_sents               = []
    pmids_counter           = Counter()
    exact_answers_counter   = Counter()
    sents_alredy_examined   = set() # i use this because during indexing i also appended the spans of the figures
    for item in res[:20]:
        doc_id          = item['pmid'].split()[0].strip()
        pmids_counter.update(Counter([doc_id]))
        par_id          = item['pmid'].split()[1].strip()
        doc_score       = item['doc_score']
        #################################################
        overall_exact   = get_answers(qtext, item['paragraph'])
        par_ex_ans_counter.update(Counter(overall_exact))
        #################################################
        for sent_score, sent_text in item['sents_with_scores']:
            if(len(sent_text)<sent_min_chars or re.sub('\s+', '', sent_text.lower()) in sents_alredy_examined):
                continue
            cand_ex_ans     = get_answers(qtext, sent_text)
            exact_answers_counter.update(Counter(cand_ex_ans))
            all_sents.append((doc_id, par_id, doc_score, overall_exact, sent_score, sent_text, cand_ex_ans))
            sents_alredy_examined.add(re.sub('\s+', '', sent_text.lower()))
    ####################################################################
    results = []
    for s in all_sents:
        # we use the jpdrmm score of the sent
        sent_score = s[4]
        # we use the jpdrmm score of the paragraph
        sent_score = sent_score * s[2]
        # # if multiple paragraphs from a document are present we boost it
        # sent_score = sent_score * pmids_counter[s[0]]
        # # if the sentence has an exact answer we boost it
        # sent_score =  sent_score * (2.0 if len(s[6])>0 else 1.0)
        # # if the exact answer of the sentence could be found in the exact answers of the paragraph we boost it
        # sent_score =  sent_score * (2.0 if any(ea in s[3] for ea in s[6]) else 1.0)
        # # if the exact answer of the sentence could be found in the top 5 exact answers of all paragraphs we boost it
        # sent_score =  sent_score * (2.0 if any(ea in [tt[0] for tt in par_ex_ans_counter.most_common(5)] for ea in s[6]) else 1.0)
        results.append((s[0] +' '  + s[1], s[5], sent_score))
    ####################################################################
    print('')
    print(qtext)
    # pprint(list(pmids_counter.most_common(5)))
    pprint(exact_answers_counter.most_common(5))
    pprint(par_ex_ans_counter.most_common(5))
    ####################################################################
    results = sorted(results, key=lambda s: s[2], reverse=True)
    ############################################
    kept    = []
    ccc     = Counter()
    for d_, s_, sc_ in results:
        if(ccc[d_]==2):
            continue
        kept.append((d_, sc_, s_))
        ccc.update(Counter([(d_)]))
        if(sum(ccc.values()) == 10):
            break
    print(40 * '=')
    pprint(flattened(list(get_answers(qtext, sent_text[2]) for sent_text in kept)))
    ####################################################################

'''
for question in tqdm(d['questions']):
    qtype           = question['type']
    if(qtype != 'factoid' and qtype != 'list'):
        continue
    qtext                   = question['body']
    res                     = retrieve_given_question(qtext)
    ###############################################################################################################
    par_ex_ans_counter      = Counter()
    all_sents               = []
    pmids_counter           = Counter()
    exact_answers_counter   = Counter()
    sents_alredy_examined   = set() # i use this because during indexing i also appended the spans of the figures
    for item in res[:20]:
        doc_id          = item['pmid'].split()[0].strip()
        pmids_counter.update(Counter([doc_id]))
        par_id          = item['pmid'].split()[1].strip()
        doc_score       = item['doc_score']
        #################################################
        overall_exact   = emit_exact_answers(qtext, item['paragraph'])
        overall_exact   = [
            t
            for t in overall_exact
            if (
                t[1] >= 0.5 and
                t[2] >= 0.5 and
                all(nono != t[0] for nono in nonos) and
                all(nono2 not in t[0] for nono2 in nonos_2)
            )
        ]
        overall_exact   = [t[0] for t in overall_exact]
        par_ex_ans_counter.update(Counter(overall_exact))
        #################################################
        for sent_score, sent_text in item['sents_with_scores']:
            if(len(sent_text)<sent_min_chars or re.sub('\s+', '', sent_text.lower()) in sents_alredy_examined):
                continue
            cand_ex_ans     = emit_exact_answers(qtext, sent_text)
            cand_ex_ans     = [
                t
                for t in cand_ex_ans
                if(
                    t[1]>=0.5 and
                    t[2]>=0.5 and
                    all(nono != t[0] for nono in nonos) and
                    all(nono2 not in t[0] for nono2 in nonos_2)
                )
            ]
            cand_ex_ans     = [t[0] for t in cand_ex_ans]
            exact_answers_counter.update(Counter(cand_ex_ans))
            all_sents.append((doc_id, par_id, doc_score, overall_exact, sent_score, sent_text, cand_ex_ans))
            sents_alredy_examined.add(re.sub('\s+', '', sent_text.lower()))
    ####################################################################
    results = []
    for s in all_sents:
        # we use the jpdrmm score of the sent
        sent_score = s[4]
        # we use the jpdrmm score of the paragraph
        sent_score = sent_score * s[2]
        # # if multiple paragraphs from a document are present we boost it
        # sent_score = sent_score * pmids_counter[s[0]]
        # # if the sentence has an exact answer we boost it
        # sent_score =  sent_score * (2.0 if len(s[6])>0 else 1.0)
        # # if the exact answer of the sentence could be found in the exact answers of the paragraph we boost it
        # sent_score =  sent_score * (2.0 if any(ea in s[3] for ea in s[6]) else 1.0)
        # # if the exact answer of the sentence could be found in the top 5 exact answers of all paragraphs we boost it
        # sent_score =  sent_score * (2.0 if any(ea in [tt[0] for tt in par_ex_ans_counter.most_common(5)] for ea in s[6]) else 1.0)
        results.append((s[0] +' '  + s[1], s[5], sent_score))
    ####################################################################
    print('')
    print(qtext)
    # pprint(list(pmids_counter.most_common(5)))
    pprint(exact_answers_counter.most_common(5))
    pprint(par_ex_ans_counter.most_common(5))
    ####################################################################
    results = sorted(results, key=lambda s: s[2], reverse=True)
    ccc     = Counter()
    for d_, s_, sc_ in results:
        if(ccc[d_]==2):
            continue
        print((d_, sc_))
        print(s_)
        ccc.update(Counter([(d_)]))
        if(sum(ccc.values()) == 10):
            break
    print(40 * '=')
    ####################################################################

'''
