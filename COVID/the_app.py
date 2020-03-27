
from retrieve_and_rerank import retrieve_given_question, pprint
quest   = 'Is coronavirus related to pneumonia ?'
results = retrieve_given_question(quest, n=100, max_year=2021)
pprint(results[0])


