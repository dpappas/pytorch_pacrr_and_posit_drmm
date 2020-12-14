
import  json
from    collections import Counter
from    pprint import pprint
from    retrieve_and_rerank import retrieve_given_question
from    emit_exact_answers import emit_exact_answers
from tqdm import tqdm

# fpath   = '/home/dpappas/BioASQ-taskSynergy-dryRun-testset'
fpath   = '/home/dpappas/COVID_SYNERGY/BioASQ-taskSynergy-testset1'
d       = json.load(open(fpath))

for question in tqdm(d['questions']):
    qtype           = question['type']
    qtext           = question['body']
    res             = retrieve_given_question(qtext)
    all_sents       = []
    pmids_counter           = Counter()
    exact_answers_counter   = Counter()
    for item in res[:10]:
        doc_id          = item['pmid'].split()[0].strip()
        pmids_counter.update(Counter([doc_id]))
        par_id          = item['pmid'].split()[1].strip()
        doc_score       = item['doc_score']
        #################################################
        overall_exact   = emit_exact_answers(qtext, item['paragraph'])
        overall_exact   = [t for t in overall_exact if (t[1] >= 0.5 and t[2] >= 0.5 and 'sars - cov' not in t[0] and 'cov - 2' not in t[0] and 'covid - 19' not in t[0] and 'coronavirus' not in t[0])]
        overall_exact   = [t[0] for t in overall_exact]
        #################################################
        for sent_score, sent_text in item['sents_with_scores']:
            cand_ex_ans     = emit_exact_answers(qtext, sent_text)
            cand_ex_ans     = [t for t in cand_ex_ans if(t[1]>=0.5 and t[2]>=0.5 and 'sars - cov' not in t[0] and 'cov - 2' not in t[0] and 'covid - 19' not in t[0] and 'coronavirus' not in t[0])]
            cand_ex_ans     = [t[0] for t in cand_ex_ans]
            exact_answers_counter.update(Counter(cand_ex_ans))
            all_sents.append((doc_id, par_id, doc_score, overall_exact, sent_score, sent_text, cand_ex_ans))
    ####################################################################
    print('')
    print(qtext)
    pprint(list(pmids_counter.most_common(5)))
    pprint(list(exact_answers_counter.most_common(5)))
    ####################################################################




