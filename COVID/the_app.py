
from retrieve_and_rerank import retrieve_given_question, pprint

quest   = 'A pneumonia outbreak associated with a new coronavirus of probable bat origin'

results = retrieve_given_question(quest, n=100, max_year=2021)
pprint(results[:2])


