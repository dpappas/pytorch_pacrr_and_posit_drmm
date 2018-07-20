#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
from flask import redirect, url_for, jsonify
from flask_cors import CORS, cross_origin
from gensim.models.keyedvectors import KeyedVectors

w2v_bin_path    = '/home/DATA/Biomedical/other/BiomedicalWordEmbeddings/binary/biomedical-vectors-200.bin'
wv              = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)

def get_vecs(words):
    wds, vecs  = [], []
    for w in words:
      if w in wv:
        vec = wv[w]
        vecs.append(vec)
        wds.append(w)
    return wds, vecs

app     = Flask(__name__)
cors    = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/get_bioasq_w2v_embeds', methods=['GET', 'POST'])
@cross_origin()
def hello():
    try:
        app.logger.debug('JOSN received')
        app.logger.debug(request.json)
        if(request.json):
            mydata = request.json
            if('tokens' in mydata):
                ret = {
                    'embeds': get_vecs(mydata['tokens'])
                }
            else:
                ret = {}
            return jsonify(ret)
    except Exception as e:
        app.logger.debug(e.message)

if __name__ == '__main__':
    app.run(port=1234, debug=True)

'''
import urllib2, json
from pprint import pprint
data    = {'tokens' : ['hello darkness my old friend ...'.split()]}
req     = urllib2.Request('http://127.0.0.1:1234/get_bioasq_w2v_embeds')
req.add_header('Content-Type', 'applications/json')
response = urllib2.urlopen(req, json.dumps(data))
pprint(response)
'''


