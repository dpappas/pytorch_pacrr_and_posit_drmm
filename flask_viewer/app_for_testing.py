
from colour import Color
from flask import Flask
from flask import render_template
from flask import request
import random
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

print('loading w2v')
w2v_bin_path    = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
wv              = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
wv              = dict([(word, wv[word]) for word in wv.vocab.keys()])

white           = Color("white")
yellow_colors   = list(white.range_to(Color("yellow"), 100))
yellow_colors   = [c.get_hex_l() for c in yellow_colors]
blue_colors     = list(white.range_to(Color("blue"), 100))
blue_colors     = [c.get_hex_l() for c in blue_colors]

app = Flask(__name__)

def get_embeds(tokens, wv):
    ret1, ret2 = [], []
    for tok in tokens:
        if(tok in wv):
            ret1.append(tok)
            ret2.append(wv[tok])
        else:
            ret1.append('UNK')
            ret2.append(wv['unk'])
    return ret1, np.array(ret2, 'float64')

def create_table(tokens1, tokens2, scores):
    ret_html = '''
    <table>
    '''
    ret_html += '<tr><td></td>'
    ####################
    for tok1 in tokens1:
        ret_html += '<th>{}</th>'.format(tok1)
    ret_html += '</tr>'
    ####################
    for i in range(len(tokens2)):
        tok2        = tokens2[i]
        ret_html    += '<tr>'
        ret_html    += '<th>{}</th>'.format(tok2)
        for j in range(len(tokens1)):
            tok1    = tokens1[j]
            score   = scores[j][i]
            ret_html += '<td title="{}" score="{}" bgcolor="{}"></div></td>'.format('{} : {} : {}'.format(tok1,tok2,str(score)), score, yellow_colors[score])
        ret_html += '</tr>'
    ret_html += '</table>'
    return ret_html

@app.route("/test_similarity_matrix", methods=["POST", "GET"])
def test_similarity_matrix():
    sent1           = 'this is the first sentence'
    sent2           = 'the second sentence which is different than the first one'
    tokens1, emb1   = get_embeds(sent1.split(), wv)
    tokens2, emb2   = get_embeds(sent2.split(), wv)
    scores          = cosine_similarity(emb1, emb2)
    #############
    ret_html    = '''
    <html>
    <head>
    <style>
    table, th, td {border: 1px solid black;}
    </style>
    </head>
    <body>
    '''
    ret_html    += create_table(tokens1, tokens2, scores)
    ret_html    += '''
    <p><b>Note:</b> Attention scores between the sentences:</p>
    <p>Sentence1: {}</p>
    <p>Sentence2: {}</p>
    </body>
    </html>
    '''.format(sent1, sent2)
    return ret_html

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
