
# from retrieve_and_rerank import retrieve_given_question, pprint
# quest   = 'Is coronavirus related to pneumonia ?'
# results = retrieve_given_question(quest, n=100, max_year=2021)
# pprint(results[0])

from retrieve_and_rerank import retrieve_given_question, get_from_id
from sklearn.preprocessing import MinMaxScaler
from colour import Color
from flask import url_for
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from collections import OrderedDict
from pprint import pprint
from rule_matching import do_for_sent
# from flask_cors import CORS
# from flask_sslify import SSLify

# pip3.6 install -U flask-cors
# pip3.6  install Flask-SSLify

white           = Color("white")
yellow_colors   = list(white.range_to(Color("yellow"), 101))
yellow_colors   = [c.get_hex_l() for c in yellow_colors]
blue_colors     = list(white.range_to(Color("blue"), 101))
blue_colors     = [c.get_hex_l() for c in blue_colors]
green_colors    = list(white.range_to(Color("green"), 101))
green_colors    = [c.get_hex_l() for c in green_colors]

app = Flask(__name__)
# CORS(app)
# sslify = SSLify(app)

@app.route("/")
def get_news():
    return render_template("home.html")

r1 = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <link href="https://unpkg.com/tailwindcss@^1.0/dist/tailwind.min.css" rel="stylesheet">
    <link rel="shortcut icon" href="static/aueb_favicon.png">

    <title>Results</title>

    <style>
    .accordion {background-color: #eee; color: #444; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px; transition: 0.4s;}
    .active, .accordion:hover {background-color: #ccc;}
    .panel {padding: 0 18px; display: none; background-color: white; overflow: hidden;}
    </style>
</head>
<body class="flex flex-wrap">
    <div class="w-full mb-6 p-4 mx-auto bg-indigo-600">
        <h4 class="text-white text-center text-2xl">AUEB NLP Group COVID-19 Search Engine</h4>
    </div>
<div class="container my-6 p-4 mx-auto">
    <div class="lg:w-3/3 md:w-3/3 mx-auto">
'''

r2 = '''
    </div>
</div>
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

@app.route("/just_the_json", methods=["POST", "GET"])
@cross_origin()
def just_the_json():
    try:
        req = request.get_json()
        print(30 * '-')
        print('request:')
        pprint(req)
        print(30 * '-')
        if(req is not None):
            question_text   = req['question']
            section         = req['section']
            min_year        = req['min_year'] if 'min_year' in req else '1600'
            max_year        = req['max_year'] if 'max_year' in req else '3000'
            if(len(section.strip())==0):
                section = None
            ###############################################################################################
            ret             = {
                'request': {
                    'question'  : question_text,
                    'section'   : section,
                    'date_from' : '',
                    'date_to'   : '',
                },
                'results': {
                    'total' : None,
                    'docs'  : []
                }
            }
            ret_dummy       = retrieve_given_question(question_text, n=20, section=section, min_year=min_year, max_year=max_year)
            ret['results']['total'] = len(ret_dummy)
            if(len(ret_dummy)==0):
                response = jsonify(ret)
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response
            scaler          = MinMaxScaler(feature_range=(0, 0.5))
            scaler.fit(np.array([d['doc_score'] for d in ret_dummy]).reshape(-1, 1))
            ###############################################################################################
            top_10_snips = []
            for doc in ret_dummy:
                top_10_snips.extend(
                    sorted(doc['sents_with_scores'],key=lambda x: x[0], reverse=True)[:2]
                )
            top_10_snips.sort(key=lambda x: x[0], reverse=True)
            top_10_snips = [t[1] for t in top_10_snips[:5]]
            ###############################################################################################
            ret['results']['total'] = len(ret_dummy)
            ###############################################################################################
            for doc in ret_dummy:
                doc_date        = doc['date']
                doc_score       = scaler.transform([[doc['doc_score']]])[0][0] + 0.5
                ###################################################################
                doc_datum = {
                    'title'     : doc['title'].strip(),
                    'doc_score' : str(doc_score * 100),
                    'section'   : section,
                    'doc_date'  : doc_date,
                    'pmid'      : None,
                    'pmcid'     : None,
                    'doi'       : None,
                    'sentences' : []
                }
                ##############################################################################################################################
                if('pmid' in doc and len(doc['pmid'].strip())!=0):
                    doc_datum['pmid'] = doc['pmid'].strip()
                if ('pmcid' in doc and len(doc['pmcid'].strip()) != 0):
                    doc_datum['pmcid'] = doc['pmcid'].strip()
                if ('doi' in doc and len(doc['doi'].strip()) != 0):
                    doc_datum['doi'] = doc['doi'].strip()
                ##############################################################################################################################
                sents_max_score = max(sent_score for sent_score, _ in doc['sents_with_scores'])
                print(sents_max_score)
                for sent_score, sent_text in doc['sents_with_scores']:
                    sent_text   = sent_text.replace('</', '< ')
                    if(sent_score == sents_max_score and sent_score >= 0.15):
                        sent_score = 1
                    if(sent_text in top_10_snips):
                        sent_score = 1
                    if(sent_score<0.45):
                        sent_score = 0.0
                    doc_datum['sentences'].append((sent_score, sent_text, do_for_sent(sent_text)))
                ret['results']['docs'].append(doc_datum)
            response = jsonify(ret)
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
        else:
            ret = {
                'request': {
                    'question'  : '',
                    'section'   : '',
                    'date_from' : '',
                    'date_to'   : '',
                },
                'error': 'empty request'
            }
            response = jsonify(ret)
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
    except Exception as ex:
        ret = {
            'request': {
                'question'  : '',
                'section'   : '',
                'date_from' : '',
                'date_to'   : '',
            },
            'error': 'Error: {}'.format(str(ex))
        }
        response = jsonify(ret)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

@app.route("/get_using_id", methods=["POST", "GET"])
@cross_origin()
def get_using_id():
    try:
        req = request.get_json()
        print(30 * '-')
        print('request:')
        pprint(req)
        print(30 * '-')
        if(req is not None):
            item_id         = req['id']
            ###############################################################################################
            ret_dummy       = get_from_id(item_id)
            if(len(ret_dummy) == 0):
                response = jsonify({})
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response
            else:
                ret_dummy = ret_dummy[0]
                del(ret_dummy['doc_vec_scibert'])
                ret = {
                    "date"          : ret_dummy['date'],
                    "doi"           : ret_dummy['doi'],
                    "joint_text"    : ret_dummy['joint_text'],
                    "pmcid"         : ret_dummy['pmcid'],
                    "pmid"          : ret_dummy['pmid'],
                    "section"       : ret_dummy['section']
                }
                ret             = ret_dummy
                # ret             = {
                #     'request': {'id'  : item_id},
                #     'results': {
                #         'total' : None,
                #         'docs'  : []
                #     }
                # }
                # ret['results']['total'] = len(ret_dummy)
                # ret['results']['docs']  = ret_dummy
                response = jsonify(ret)
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response
        else:
            ret = {
                'request': {},
                'error': 'empty request'
            }
            response = jsonify(ret)
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response
    except Exception as ex:
        ret = {
            'request': {},
            'error': 'Error: {}'.format(str(ex))
        }
        response = jsonify(ret)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

@app.route("/submit_question", methods=["POST", "GET"])
@cross_origin()
def submit_question():
    question_text   = request.form.get("sent1") #.strip()
    section         = request.form.get("section") #.strip()
    if(len(section.strip())==0):
        section = None
    ###############################################################################################
    # if(question_text is None or len(question_text)==0):
    #     question_text = 'covid-19'
    print(question_text)
    print(section)
    ###############################################################################################
    text_to_return  = r1 + '\n' # + r2
    text_to_return  += '<h2 class="block uppercase tracking-wide text-gray-700 text-base font-normal mb-2">' \
                       'Results for the question: {}</h2>'.format(question_text) + '\n'
    ret_dummy       = retrieve_given_question(question_text, n=20, section=section)
    if(len(ret_dummy)==0):
        text_to_return += '\n' + r2
        return text_to_return
    ###############################################################################################
    scaler          = MinMaxScaler(feature_range=(0, 0.5))
    scaler.fit(np.array([d['doc_score'] for d in ret_dummy]).reshape(-1, 1))
    ###############################################################################################
    top_10_snips = []
    for doc in ret_dummy:
        top_10_snips.extend(
            sorted(doc['sents_with_scores'],key=lambda x: x[0], reverse=True)[:2]
        )
    top_10_snips.sort(key=lambda x: x[0], reverse=True)
    top_10_snips = [t[1] for t in top_10_snips[:5]]
    ###############################################################################################
    for doc in ret_dummy:
        doc_date        = doc['date']
        doc_score       = scaler.transform([[doc['doc_score']]])[0][0] + 0.5
        doc_bgcolor     = green_colors[int(doc_score * 100)]
        doc_txtcolor    = 'white' if (doc_score > 0.5) else 'black'
        # text_to_return  += '<button title="{}" class="accordion" style="background-color:{};color:{};">PMID:{}   ||   PMCID:{}   ||   doi:{}   ||   SECTION:{}   ||   Date:{}</button><div class="panel">'.format(
        #     str(doc_score * 100), doc_bgcolor, doc_txtcolor, doc['pmid'], doc['pmcid'], doc['doi'], doc['section'], doc_date
        # )
        text_to_return  += '<button title="{}" class="accordion" style="background-color:{};color:{};">' \
                           '{}</button><div class="panel m-4">'.format(
            str(doc_score * 100),
            doc_bgcolor,
            doc_txtcolor,
            'Title: '+ (doc['title'] if(len(doc['title'])) else '(Not Available)')
        )
        text_to_return += '<div class="my-1" title="{}" style="width:100%;background-color:{};">Date: {}  ||  Section: {}</div>'.format(
            doc['section'], '#e2e8f0', doc_date, doc['section']
        )
        ##############################################################################################################################
        if('pmid' in doc and len(doc['pmid'].strip())!=0):
            text_to_return += '<div title="{}" class="my-1" style="width:100%;background-color:{};">{}</div>'.format(
                'link',
                '#e2e8f0',
                'Available on: <a href="https://www.ncbi.nlm.nih.gov/pubmed/?term={}">PMID: {}</a>'.format(doc['pmid'], doc['pmid'])
            )
        if ('pmcid' in doc and len(doc['pmcid'].strip()) != 0):
            text_to_return += '<div title="{}" class="my-1" class="my-1" style="width:100%;background-color:{};">{}</div>'.format(
                'link',
                '#e2e8f0',
                'Available on: <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/{}">PMC: {}</a>'.format(doc['pmcid'], doc['pmcid'])
            )
        if ('doi' in doc and len(doc['doi'].strip()) != 0):
            text_to_return += '<div title="{}" class="my-1" style="width:100%;background-color:{};">{}</div>'.format(
                'link',
                '#e2e8f0',
                'Available on: <a href="http://doi.org/{}">Doi : {}</a>'.format(doc['doi'], doc['doi'])
            )
        ##############################################################################################################################
        text_to_return += '<div title="" style="width:100%;height:20px;background-color:white;"> </div>'
        sents_max_score = max(sent_score for sent_score, _ in doc['sents_with_scores'])
        print(sents_max_score)
        for sent_score, sent_text in doc['sents_with_scores']:
            sent_text               = sent_text.replace('</', '< ')
            if(sent_score == sents_max_score and sent_score >= 0.15):
                sent_score = 1
            if(sent_text in top_10_snips):
                sent_score = 1
            if(sent_score<0.45):
                sent_score = 0.0
            text_to_return += '<div title="{}" class="my-1" style="width:100%;background-color:{};">{}</div>'.format(
                sent_score, yellow_colors[int(sent_score*100)], sent_text)
        ##############################################################################################################################
        text_to_return += '</div>'
    text_to_return += '\n' + r2
    return text_to_return

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    # app.add_url_rule('/favicon.ico', redirect_to=url_for('static', filename='aueb_favicon.png'))
    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, ssl_context='adhoc')
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

