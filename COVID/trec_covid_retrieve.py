
import json
from pprint import pprint
from retrieve_and_rerank import retrieve_given_question

mappings            = json.load(open())

question_text1      = 'what is the origin of COVID-19'
question_text2      = "seeking range of information about the SARS-CoV-2 virus's origin, including its evolution, animal source, and first transmission into humans"

ret_dummy1          = retrieve_given_question(question_text1, n=100, section=None)
ret_dummy2          = retrieve_given_question(question_text2, n=100, section=None)

pprint(ret_dummy1[0])
