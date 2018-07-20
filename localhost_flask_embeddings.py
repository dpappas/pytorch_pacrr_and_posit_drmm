#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
from flask import redirect, url_for, jsonify
from gensim.models.keyedvectors import KeyedVectors

app = Flask(__name__)

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

@app.route('/get_bioasq_w2v_embeds', methods=['GET', 'POST'])
def hello():
    try:
        app.logger.debug('JOSN received')
        app.logger.debug(request.json)
        if(request.json):
            mydata = request.json
            if( 'tokens' in mydata):
                return jsonify({'embeds':get_vecs(mydata['tokens'])})
            else:
                return jsonify({})
    except Exception as e:
        app.logger.debug(e.message)

if __name__ == '__main__':
    app.run(port=1234, debug=True)

