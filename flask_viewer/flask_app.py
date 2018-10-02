
from colour import Color
from    nltk.tokenize import sent_tokenize
from flask import Flask
from flask import render_template
from flask import request
import random

red     = Color("white")
colors  = list(red.range_to(Color("yellow"), 100))
colors  = [c.get_hex_l() for c in colors]

app = Flask(__name__)

@app.route("/")
def get_news():
    return render_template("home.html")

@app.route("/submit_question", methods=["POST"])
def get_quest_results():
    q = request.form.get("the_quest")
    d = request.form.get("the_doc")
    m = request.form.get("the_mesh")
    m = [t.strip() for t in m.split('||')]
    # return render_template("home.html")
    # add the question as header
    ret_html = '<h2>Question: </h2>'
    ret_html += '<p>{}</p>'.format(q)
    ret_html += '</br></br>'
    # add the scored sentences
    ret_html += '<h2>Document:</h2>'
    for sent in sent_tokenize(d):
        score = random.randint(1,100)
        ret_html += '<div style="background-color:{}">{}</div>'.format(colors[score], sent)
    ret_html += '</br></br>'
    # add the scored mesh terms
    ret_html += '<h2>Mesh Terms:</h2>'
    for sent in m:
        score = random.randint(1,100)
        ret_html += '<div style="background-color:{}">{}</div>'.format(colors[score], sent)
    ret_html += '</br></br>'
    return ret_html

if __name__ == '__main__':
    app.run(port=5000, debug=True)
