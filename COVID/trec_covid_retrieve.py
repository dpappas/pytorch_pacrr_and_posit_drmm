
import json
from pprint import pprint
from bs4 import BeautifulSoup
from retrieve_and_rerank import retrieve_given_question
from tqdm import tqdm

mappings_1          = json.load(open('C:/Users/dvpap/Downloads/TREC-COVID/batch1/cord_uid_mappings.json'))
mappings_2          = json.load(open('C:/Users/dvpap/Downloads/TREC-COVID/batch2/cord_uid_mappings.json'))

mappings = {
    'dois'      : {},
    'pmcids'    : {},
    'pmids'     : {},
}
for k in mappings_2:
    for item in mappings_2[k].items():
        if(not item[0] in mappings_1[k]):
            mappings[k][item[0]] = item[1]

soup                = BeautifulSoup(open('topics-rnd1.xml').read(), "lxml")
run_tag             = 'AUEB_NLP_GROUP'

fp                  = open('trec_results.txt', 'w')
for topic in tqdm(soup.find_all('topic')):
    topicid     = topic.get('number')
    query       = topic.find('query').text.strip()
    question    = topic.find('question').text.strip()
    narrative   = topic.find('narrative').text.strip()
    ret_dummy   = retrieve_given_question(query + ' ' + question + ' '+narrative, n=5000, section=None)
    ######################################################################################################
    aggregated_results  = {}
    for item in ret_dummy:
        docid = None
        if(not docid and item['doi'] in mappings['dois']):
            docid = mappings['dois'][item['doi']]
        if(not docid and item['pmcid'] in mappings['pmcids']):
            docid = mappings['pmcids'][item['pmcid']]
        if (not docid and item['pmid'] in mappings['pmids']):
            docid = mappings['pmids'][item['pmid']]
        if(docid):
            aggregated_results[docid] = item['doc_score']
    ######################################################################################################
    aggregated_results = sorted(aggregated_results.items(), key=lambda x: x[1], reverse=True)[:1000]
    ######################################################################################################
    for rank, (docid,score)  in enumerate(aggregated_results):
        fp.write('{} {} {} {} {} {}\n'.format(topicid, 'Q0', docid, rank, score, run_tag))

fp.close()

'''
Each run must be contained in a single text file. Each line in the file must be in the form

topicid Q0 docid rank score run-tag

where
topicid	is the topic number (1..30)
Q0	is the literal 'Q0' (currently unused, but trec_eval expects this column to be present)
docid	is the cord_uid of the document retrieved in this position
rank	is the rank position of this document in the list
score	is the similarity score computed by the system for this document. When your run is processed (to create judgment sets and to score it using trec_eval), the run will be sorted by decreasing score and the assigned ranks will be ignored. In particular, trec_eval will sort documents with tied scores in an arbitrary order. If you want the precise ranking you submit to be used, that ranking must be reflected in the assigned scores, not the ranks.
run-tag	is a name assigned to the run. Tags must be unique across both your own runs and all other participants' runs, so you may have to choose a new tag if the one you used is already taken. Tags are strings of no more than 20 characters and may use letters, numbers, underscore (_), hyphen (-), and period (.) only. It will be best if you make the run tag semantically meaningful to identify you (e.g., if your team ID is NIST, then 'NIST-prise-tfidf' is a much better run tag than 'run1'). Every line in the submission file must end with the same run-tag.

'''


