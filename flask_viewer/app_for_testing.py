
from colour import Color
from flask import Flask
from flask import render_template
from flask import request
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re
import random
import numpy as np
import pickle

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
softmax     = lambda z: np.exp(z) / np.sum(np.exp(z))

def idf_val(w, idf, max_idf):
    if w in idf:
        return idf[w]
    return max_idf

def tokenize(x):
  return bioclean(x)

def load_idfs(idf_path):
    with open(idf_path, 'rb') as f:
        idf = pickle.load(f)
    ret = {}
    for w in idf:
        ret[w] = idf[w]
    max_idf = 0.0
    for w in idf:
        if idf[w] > max_idf:
            max_idf = idf[w]
    idf = None
    print('Loaded idf tables with max idf {}'.format(max_idf))
    return ret, max_idf

#################
idf_pickle_path = "C:\\Users\\dvpap\\Downloads\\bioasq_all\\idf.pkl"
w2v_bin_path    = "C:\\Users\\dvpap\\Downloads\\bioasq_all\\pubmed2018_w2v_30D.bin"
#################
idf, max_idf    = load_idfs(idf_pickle_path)
print('loading w2v')
wv              = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
wv              = dict([(word, wv[word]) for word in wv.vocab.keys()])
#################

white           = Color("white")
yellow_colors   = list(white.range_to(Color("yellow"), 101))
yellow_colors   = [c.get_hex_l() for c in yellow_colors]
blue_colors     = list(white.range_to(Color("blue"), 101))
blue_colors     = [c.get_hex_l() for c in blue_colors]

app = Flask(__name__)

def create_one_hot_and_sim(tokens1, tokens2):
    '''
    :param tokens1:
    :param tokens2:
    :return:
    exxample call : create_one_hot_and_sim('c d e'.split(), 'a b c'.split())
    '''
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    #
    values = list(set(tokens1 + tokens2))
    integer_encoded = label_encoder.fit_transform(values)
    #
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)
    #
    lab1 = label_encoder.transform(tokens1)
    lab1 = np.expand_dims(lab1, axis=1)
    oh1 = onehot_encoder.transform(lab1)
    #
    lab2 = label_encoder.transform(tokens2)
    lab2 = np.expand_dims(lab2, axis=1)
    oh2 = onehot_encoder.transform(lab2)
    #
    ret = np.matmul(oh1, np.transpose(oh2), out=None)
    #
    return oh1, oh2, ret

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
    ret_html = '<table>'
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
            score   = int(scores[j][i])
            ret_html += '<td style="min-width:60px" title="{}" score="{}" bgcolor="{}"></div></td>'.format('{} : {} : {}'.format(tok1,tok2,str(score)), score, yellow_colors[score])
        ret_html += '</tr>'
    ret_html += '</table>'
    ####################
    max_scores      = scores.max(axis=1)
    aver5_scores    = np.average(np.sort(scores, axis=1)[:, -5:], axis=1)
    # print(max_scores.shape)
    ret_html        += "<p><b>Pooled scores:</b></p>"
    ret_html        += '<table>'
    ret_html        += '<tr><td></td>'
    for tok1 in tokens1:
        ret_html    += '<th>{}</th>'.format(tok1)
    ret_html        += '</tr>'
    ####################
    ret_html        += '<tr><td>Max Score:</td>'
    for j in range(len(tokens1)):
        score       =  int(max_scores[j])
        ret_html    += '<td style="min-width:60px" score="{}" bgcolor="{}"></div></td>'.format(score, yellow_colors[score])
    ####################
    ret_html        += '<tr><td>Aver of max 5 Scores:</td>'
    for j in range(len(tokens1)):
        score       =  int(aver5_scores[j])
        ret_html    += '<td style="min-width:60px" score="{}" bgcolor="{}"></div></td>'.format(score, yellow_colors[score])
    ####################
    ret_html        += '<tr><td>normalized (divided by max) IDF Score:</td>'
    for j in range(len(tokens1)):
        score       = idf_val(tokens1[j], idf, max_idf)
        score       = int((score/max_idf) * 100)
        ret_html    += '<td style="min-width:60px" score="{}" bgcolor="{}"></div></td>'.format(score, blue_colors[score])
    ####################
    ret_html     += '</tr>'
    ret_html     += '</table>'
    return ret_html

@app.route("/")
def get_news():
    return render_template("sentence_similarity.html")

@app.route("/test_similarity_matrix", methods=["POST", "GET"])
def test_similarity_matrix():
    sent1           = request.form.get("sent1").strip()
    sent2           = request.form.get("sent2").strip()
    tokens1         = tokenize(sent1)
    tokens2         = tokenize(sent2)
    tokens1, emb1   = get_embeds(tokens1, wv)
    tokens2, emb2   = get_embeds(tokens2, wv)
    scores          = cosine_similarity(emb1, emb2) * 100
    _, _, scores_2  = create_one_hot_and_sim(tokens1, tokens2)
    print(scores_2.shape)
    #############
    ret_html    = '''
    <html>
    <head>
    <style>
    table, th, td {border: 1px solid black;}
    .floatLeft { width: 50%; float: left; }
    .floatRight {width: 50%; float: right; }
    .container { overflow: hidden; }
    </style>
    </head>
    <body>
    '''
    ret_html    += '''
    <div class="container">
    <div class="floatLeft">
    '''
    ret_html    += create_table(tokens1, tokens2, scores)
    ret_html    += '</div>'
    ret_html    += '<div class="floatRight">'
    ret_html    += '''
    <p><b>Note:</b> Attention scores between the sentences:</p>
    <p>Sentence1: {}</p>
    <p>Sentence2: {}</p>
    </body>
    </html>
    '''.format(sent1, sent2)
    ret_html    += '</div></div>'
    return ret_html

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


