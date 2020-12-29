
import  json, re
from    collections import Counter
from    pprint import pprint
from    retrieve_and_rerank import retrieve_given_question
from    retrieve_docs       import get_noun_chunks
from    emit_exact_answers  import emit_exact_answers
from    tqdm import tqdm

'''
from    textacy import make_spacy_doc, keyterms

bioclean_mod    = lambda t: re.sub('[~`@#$-=<>/.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').replace("\n", ' ').strip().lower())
import spacy
nlp 	= spacy.load("en_core_sci_lg")

def get_keyphrases_sgrank(text):
    # doc = make_spacy_doc(bioclean_mod(text), lang='en')
    doc = nlp(text)
    keyphrases = keyterms.sgrank(
        doc,
        ngrams       = tuple(range(1, 4)),
        normalize    = None,  # None, # u'lemma', # u'lower'
        window_width = 50,
        n_keyterms   = 5,
        idf          = None,
        include_pos  = ("NOUN", "PROPN", "ADJ"),  # ("NOUN", "PROPN", "ADJ"), # ("NOUN", "PROPN", "ADJ", "VERB", "CCONJ"),
    )
    keyphrases = [t[0] for t in keyphrases[0]]
    return keyphrases

from elasticsearch import Elasticsearch
es          = Elasticsearch('127.0.0.1', verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)
doc_index   = 'allenai_covid_index_2020_11_29_csv'
'''

flattened = lambda l: [item for sublist in l for item in sublist]

# # fpath   = '/home/dpappas/BioASQ-taskSynergy-dryRun-testset'

# feedback_fpath  = None
# fpath           = '/home/dpappas/COVID_SYNERGY/BioASQ-taskSynergy-testset1'
# opath           = '/home/dpappas/COVID_SYNERGY/BioASQ-taskSynergy-testset1_ouputs.json'
# d               = json.load(open(fpath))

feedback_fpath  = '/home/dpappas/COVID_SYNERGY/BioASQ-taskSynergy-feedback_round2.json'
fpath           = '/home/dpappas/COVID_SYNERGY/BioASQ-taskSynergy-testset2.json'
opath           = '/home/dpappas/COVID_SYNERGY/BioASQ-taskSynergy-testset2_ouputs.json'
d               = json.load(open(fpath))

feedback            = json.load(open(feedback_fpath))
qtext_to_qid            = {}
# qid_to_pos_doc_chunks   = {}
qid_to_pos_chunks       = {}
qid_to_neg_chunks       = {}
qid_to_pos_snips        = {}
qid_to_neg_snips        = {}
qid_to_pos_docids       = {}
qid_to_neg_docids       = {}
for quest in tqdm(feedback['questions']):
    # print(quest.keys())
    print(quest['body'])
    qtext_to_qid[quest['body']]     = quest['id']
    #################################################
    neg_snips                       = [sn['text'] for sn in quest['snippets'] if(not sn['golden'])]
    pos_snips                       = [sn['text'] for sn in quest['snippets'] if(sn['golden'])]
    neg_chunks                      = Counter([t.lower() for t in flattened([get_noun_chunks(sn) for sn in neg_snips])])
    pos_chunks                      = Counter([t.lower() for t in flattened([get_noun_chunks(sn) for sn in pos_snips])])
    qid_to_pos_chunks[quest['id']]  = pos_chunks
    qid_to_neg_chunks[quest['id']]  = neg_chunks
    qid_to_pos_snips[quest['id']]   = pos_snips
    qid_to_neg_snips[quest['id']]   = neg_snips
    #################################################
    pos_docids                      = [sn['id'] for sn in quest['documents'] if(sn['golden'])]
    qid_to_pos_docids[quest['id']]  = pos_docids
    # es_res                          = es.mget(body={'ids': pos_docids}, index=doc_index)
    # posdocs                         = [t['_source']['joint_text'].replace('--------------------',' ') for t in es_res['docs'] if '_source' in t]
    # posdocs_chunks                  = Counter([t.lower() for t in flattened([get_keyphrases_sgrank(sn) for sn in posdocs])])
    # #################################################
    neg_docids                      = [sn['id'] for sn in quest['documents'] if(not sn['golden'])]
    qid_to_neg_docids[quest['id']]  = neg_docids
    # es_res                          = es.mget(body={'ids': neg_docids}, index=doc_index)
    # negdocs                         = [t['_source']['joint_text'].replace('--------------------',' ') for t in es_res['docs'] if '_source' in t]
    # negdocs_chunks                  = Counter([t.lower() for t in flattened([get_keyphrases_sgrank(sn) for sn in negdocs])])
    # #################################################
    # pprint(
    #     [
    #         c for c in posdocs_chunks
    #         if c not in negdocs_chunks
    #      ]
    # )
    # break


nonos           = ['sars', 'sars - cov', 'cov - 2', 'coronavirus', 'covid - 19', 'covid']
nonos_2         = ['et al', 'et. al', '>']
sent_min_chars  = 20

def get_answers(qtext, le_text):
    overall_exact   = emit_exact_answers(qtext, le_text) #
    overall_exact   = [
        t for t in overall_exact
        if (
            t[1] >= 0.5 and
            t[2] >= 0.5 and
            all(nono != t[0] for nono in nonos) and
            all(nono2 not in t[0] for nono2 in nonos_2)
        )
    ]
    overall_exact   = [t[0] for t in overall_exact]
    return overall_exact

exported = {"questions": []}

for question in tqdm(d['questions']):
    qtext                   = question['body']
    qtype                   = question['type']
    q_export                = {
        "body"          : qtext,
        "id"            : question['id'],
        "type"          : qtype,
        "documents"     : [],
        "snippets"      : [],
        "answer_ready"  : False,
        "ideal_answer"  : '',
        "exact_answer"  : []
    }
    res                     = retrieve_given_question(
        qtext,
        n               = 100,
        exclude_pmids   = qid_to_pos_docids[question['id']] + qid_to_neg_docids[question['id']]
    )
    ###############################################################################################################
    par_ex_ans_counter      = Counter()
    all_sents               = []
    pmids_counter           = Counter()
    exact_answers_counter   = Counter()
    sents_alredy_examined   = set() # i use this because during indexing i also appended the spans of the figures
    for item in res:
        # doc_id          = item['pmid'].split()[0].strip()
        doc_id          = item['cord_uid'].split()[0].strip()
        pmids_counter.update(Counter([doc_id]))
        par_id          = '' # item['pmid'].split()[1].strip()
        doc_score       = item['doc_score']
        #################################################
        if (qtype == 'factoid' or qtype == 'list'):
            overall_exact   = get_answers(qtext, item['paragraph'])
        else:
            overall_exact   = []
        par_ex_ans_counter.update(Counter(overall_exact))
        #################################################
        for sent_score, sent_text in item['sents_with_scores']:
            if(
                len(sent_text)<sent_min_chars or
                # re.sub('\s+', '', sent_text.lower()) in sents_alredy_examined
                re.sub('\W+', '', sent_text.lower()) in sents_alredy_examined
            ):
                continue
            if (qtype == 'factoid' or qtype == 'list'):
                cand_ex_ans   = get_answers(qtext, sent_text)
            else:
                cand_ex_ans   = []
            exact_answers_counter.update(Counter(cand_ex_ans))
            if sent_text in item['title']:
                section                 = 'title'
                offsetInBeginSection    = item['title'].index(sent_text)
                offsetInEndSection      = offsetInBeginSection + len(sent_text)
            else:
                section                 = 'abstract'
                offsetInBeginSection    = item['abstract'].index(sent_text)
                offsetInEndSection      = offsetInBeginSection + len(sent_text)
            all_sents.append(
                (
                    doc_id,
                    par_id,
                    doc_score,
                    overall_exact,
                    sent_score,
                    sent_text,
                    cand_ex_ans,
                    section,
                    offsetInBeginSection,
                    offsetInEndSection,
                )
            )
            # sents_alredy_examined.add(re.sub('\s+', '', sent_text.lower()))
            sents_alredy_examined.add(re.sub('\W+', '', sent_text.lower()))
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
        results.append((s[0], s[5], sent_score, s[7], s[8], s[9]))
    ####################################################################
    # print('')
    # print(qtext)
    # # pprint(list(pmids_counter.most_common(5)))
    # pprint(exact_answers_counter.most_common(5))
    # pprint(par_ex_ans_counter.most_common(5))
    ####################################################################
    results = sorted(results, key=lambda s: s[2], reverse=True)
    ############################################
    kept    = []
    ccc     = Counter()
    for d_, s_, sc_, sec_, of1, of2 in results:
        if(d_ not in q_export['documents']):
            q_export['documents'].append(d_)
            if(len(q_export['documents'])==10):
                break
    for d_, s_, sc_, sec_, of1, of2 in results:
        if(ccc[d_]==2):
            continue
        q_export['snippets'].append(
            {
                "document"              : d_,
                "offsetInBeginSection"  : of1,
                "offsetInEndSection"    : of2,
                "text"                  : s_,
                "beginSection"          : sec_,
                "endSection"            : sec_
            }
        )
        kept.append((d_, sc_, s_))
        ccc.update(Counter([(d_)]))
        if(sum(ccc.values()) == 10):
            break
    # print('')
    # print(40 * '=')
    # print(qtext)
    # print('\n'.join([s[2] for s in kept]))
    # pprint(eas.most_common(10))
    if(qtype == 'factoid'):
        eas = Counter(flattened(list(get_answers(qtext, sent_text[2]) for sent_text in kept)))
        q_export['exact_answer'] =[[ea[0]] for ea in eas.most_common(5)]
        q_export['answer_ready'] = True
    elif(qtype == 'list'):
        eas = Counter(flattened(list(get_answers(qtext, sent_text[2]) for sent_text in kept)))
        q_export['exact_answer'] =[[ea[0]] for ea in eas.most_common(10)]
        q_export['answer_ready'] = True
    elif (qtype == 'summary'):
        for snip in q_export['snippets']:
            if(len(q_export['ideal_answer'].split() + snip['text'].split()) < 200):
                q_export['ideal_answer'] = q_export['ideal_answer'] + ' ' + snip['text']
                q_export['ideal_answer'] = q_export['ideal_answer'].strip()
        q_export['answer_ready'] = True
    elif (qtype == 'yesno'):
        q_export['answer_ready'] = True
        q_export['exact_answer'] = 'yes'
    ####################################################################
    exported["questions"].append(q_export)
    ####################################################################
    break

with open(opath, 'w') as f:
    f.write(json.dumps(exported, indent=4, sort_keys=False))


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
