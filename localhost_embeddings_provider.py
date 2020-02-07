
from flask import  Flask
from flask import request
from flask import redirect, url_for, jsonify
import traceback

from    gensim.models.keyedvectors  import KeyedVectors

print('Loading Embeddings')
w2v_bin_path    = '/home/dpappas/bioasq_all/pubmed2018_w2v_30D.bin'
wv              = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
print('Done')

app         = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/get_embeds', methods=['GET','POST'])
def get_data_using_slug():
    try:
        app.logger.debug("received...")
        token = request.args.get('token')
        ret = {'emb': wv[token]}
        app.logger.debug(ret)
        return jsonify(ret)
    except Exception as e:
        app.logger.debug(str(e))
        traceback.print_exc()
        ret = {'success': 0, 'message': str(e)+'\n'+traceback.format_exc()}
        app.logger.debug(ret)
        return jsonify(ret)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9250, debug=True, threaded=True)
    # app.run(host='localhost', port=9250, debug=True, threaded=True)

