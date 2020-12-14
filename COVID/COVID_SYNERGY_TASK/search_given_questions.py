
import  json
from    collections import Counter
from    pprint import pprint
from    retrieve_and_rerank import retrieve_given_question
from    emit_exact_answers import emit_exact_answers

fpath   = '/home/dpappas/BioASQ-taskSynergy-dryRun-testset'
d       = json.load(open(fpath))

for question in d['questions']:
    qtype   = question['type']
    qtext   = question['body']
    print(qtext)
    res = retrieve_given_question(qtext)
    pprint(res[0].keys())
    for score, sent in res[0]['sents_with_scores']:
        pprint(emit_exact_answers(qtext, sent))
    break
    # pprint(res)
    c = Counter()
    c.update(
        item['_id'].split()[0]
        for item in res['hits']['hits']
    )
    ####################################################################





