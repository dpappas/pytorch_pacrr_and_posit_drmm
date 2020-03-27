
# from retrieve_and_rerank import retrieve_given_question, pprint
# quest   = 'Is coronavirus related to pneumonia ?'
# results = retrieve_given_question(quest, n=100, max_year=2021)
# pprint(results[0])

from retrieve_and_rerank import retrieve_given_question
from sklearn.preprocessing import MinMaxScaler
from colour import Color
from flask import url_for
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from collections import OrderedDict

white           = Color("white")
yellow_colors   = list(white.range_to(Color("yellow"), 101))
yellow_colors   = [c.get_hex_l() for c in yellow_colors]
blue_colors     = list(white.range_to(Color("blue"), 101))
blue_colors     = [c.get_hex_l() for c in blue_colors]
green_colors    = list(white.range_to(Color("green"), 101))
green_colors    = [c.get_hex_l() for c in green_colors]

app = Flask(__name__)

@app.route("/")
def get_news():
    return render_template("home.html")

r1 = '''
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.accordion {background-color: #eee; color: #444; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px; transition: 0.4s;}
.active, .accordion:hover {background-color: #ccc;}
.panel {padding: 0 18px; display: none; background-color: white; overflow: hidden;}
</style>
</head>
<body>
<title>Results</title>
'''

r2 = '''

<script>
var acc = document.getElementsByClassName("accordion");
var i;

for (i = 0; i < acc.length; i++) {
  acc[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var panel = this.nextElementSibling;
    if (panel.style.display === "block") {
      panel.style.display = "none";
    } else {
      panel.style.display = "block";
    }
  });
}
</script>

</body>
</html>
'''

@app.route("/submit_question", methods=["POST", "GET"])
def submit_question():
    question_text   = request.form.get("sent1") #.strip()
    section         = request.form.get("section") #.strip()
    if(len(section.strip())==0):
        section = None
    ###############################################################################################
    print(question_text)
    print(section)
    ###############################################################################################
    text_to_return  = r1 + '\n' # + r2
    text_to_return  += '<h2>Results for the question: {}</h2>'.format(question_text) + '\n'
    ret_dummy       = retrieve_given_question(question_text, section=section)
    ###############################################################################################
    scaler          = MinMaxScaler(feature_range=(0, 0.5))
    scaler.fit(np.array([ret_dummy[doc_id]['doc_score'] for doc_id in ret_dummy]).reshape(-1, 1))
    for doc_id in ret_dummy:
        doc_date        = ret_dummy[doc_id]['doc_date']
        doc_score       = scaler.transform([[ret_dummy[doc_id]['doc_score']]])[0][0] + 0.5
        doc_bgcolor     = green_colors[int(doc_score*100)]
        doc_txtcolor    = 'white' if(doc_score>0.5) else 'black'
        text_to_return  += '<button title="{}" class="accordion" style="background-color:{};color:{};">PMID:{}    Date:{}</button><div class="panel">'.format(str(doc_score*100), doc_bgcolor, doc_txtcolor, doc_id, doc_date)
        for sent in ret_dummy[doc_id]['sentences']:
            # print(sent)
            sent_score, sent_text   = sent
            sent_text               = sent_text.replace('</', '< ')
            if(sent_score<0.45):
                sent_score = 0.0
            text_to_return += '<div title="{}" style="width:100%;background-color:{};">{}</div>'.format(sent_score, yellow_colors[int(sent_score*100)], sent_text)
        text_to_return += '<div title="{}" style="width:100%;background-color:{};">{}</div>'.format(
            'link',
            'white',
            'Available on: <a href="https://www.ncbi.nlm.nih.gov/pubmed/?term={}">{}</a>'.format(doc_id, doc_id)
        )
        text_to_return  += '</div>'
    text_to_return += '\n' + r2
    return text_to_return

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
