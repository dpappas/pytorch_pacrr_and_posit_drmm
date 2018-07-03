import re
import sys
import json
import ijson
import pickle


# Modified bioclean: also split on dashes. Works better for retrieval with galago.
bioclean_mod = lambda t: re.sub(
    '[.,?;*!%^&_+():-\[\]{}]',
    '',
    t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace("-", ' ').strip().lower()
).split()

def prerpocess(path_in, path_out):
    with open('stopwords.pkl', 'rb') as f:
        stopwords = pickle.load(f)
    #
    f_in    = open(path_in, 'r', encoding='utf-8')
    queries = ijson.items(f_in, 'questions.item')
    #
    q_array = []
    for query in queries:
        #
        tokenized_body = bioclean_mod(query['body'])
        tokenized_body = [t for t in tokenized_body if t not in stopwords]
        #
        body = ' '.join(tokenized_body)
        q_array.append({"text": body, "number": query["id"]})
    with open(path_out, 'w+') as outfile:
        outfile.write(json.dumps({"queries": [ob for ob in q_array]}, indent=4))

prerpocess(sys.argv[1], sys.argv[2])


