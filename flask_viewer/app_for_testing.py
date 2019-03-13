
from colour import Color
from flask import Flask
from flask import render_template
from flask import request
import random

white           = Color("white")
yellow_colors   = list(white.range_to(Color("yellow"), 100))
yellow_colors   = [c.get_hex_l() for c in yellow_colors]
blue_colors     = list(white.range_to(Color("blue"), 100))
blue_colors     = [c.get_hex_l() for c in blue_colors]

app = Flask(__name__)

@app.route("/test_similarity_matrix", methods=["POST", "GET"])
def test_similarity_matrix():
    tokens1 = 'this is the first sentence'.split()
    tokens2 = 'the second sentence which is different than the first one'.split()
    ret_html = '''
    <html>
    <head>
    <style>
    table, th, td {border: 1px solid black;}
    </style>
    </head>
    <body>
    <table>
    '''
    ret_html += '''
    </table>
    <p><b>Note:</b> Attention scores between two sentences.</p>
    </body>
    </html>
    '''
    return ret_html

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
