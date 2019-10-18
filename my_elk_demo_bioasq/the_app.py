
from .emit_given_text import get_results_for_one_question

from colour import Color
from flask import url_for
from flask import Flask
from flask import render_template
from flask import request

white           = Color("white")
yellow_colors   = list(white.range_to(Color("yellow"), 101))
yellow_colors   = [c.get_hex_l() for c in yellow_colors]
blue_colors     = list(white.range_to(Color("blue"), 101))
blue_colors     = [c.get_hex_l() for c in blue_colors]

app = Flask(__name__)

@app.route("/")
def get_news():
    return render_template("sentence_similarity.html")

# @app.route("/test_similarity_matrix", methods=["POST", "GET"])
# def test_similarity_matrix():
#     sent1           = request.form.get("sent1").strip()
#     sent2           = request.form.get("sent2").strip()
#     tokens1         = tokenize(sent1)
#     tokens2         = tokenize(sent2)
#     tokens1, emb1   = get_embeds(tokens1, wv)
#     tokens2, emb2   = get_embeds(tokens2, wv)
#     scores          = cosine_similarity(emb1, emb2).clip(min=0) * 100
#     # scores          = cosine_similarity(emb1, emb2)
#     # scores          = (scores + 1.0) / 2.0 # max min normalization
#     # scores          = scores * 100
#     _, _, scores_2  = create_one_hot_and_sim(tokens1, tokens2)
#     #############
#     ret_html    = '''
#     <html>
#     <head>
#     <style>
#     table, th, td {border: 1px solid black;}
#     .floatLeft { width: 50%; float: left; }
#     .floatRight {width: 50%; float: right; }
#     .container { overflow: hidden; }
#     </style>
#     </head>
#     <body>
#     '''
#     ret_html    += '<div class="container">'
#     ret_html    += '<div class="floatLeft">'
#     ret_html    += '<p><h2>W2V cosine similarity (clipped negative to zero):</h2></p>'
#     ret_html    += create_table(tokens1, tokens2, scores)
#     ret_html    += '</div>'
#     ret_html    += '<div class="floatRight">'
#     ret_html    += '<p><h2>One-Hot cosine similarity:</h2></p>'
#     ret_html    += create_table(tokens1, tokens2, scores_2*100)
#     ret_html    += '</div>'
#     ret_html    += '</div>'
#     ret_html    += '''
#     <p><b>Note:</b> Attention scores between the sentences:</p>
#     <p>Sentence1: {}</p>
#     <p>Sentence2: {}</p>
#     </body>
#     </html>
#     '''.format(sent1, sent2)
#     return ret_html

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


