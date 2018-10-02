
from colour import Color
red     = Color("white")
colors  = list(red.range_to(Color("yellow"), 100))
colors  = [c.get_hex_l() for c in colors]

# print(colors)
# exit()

from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__)

@app.route("/")
def get_news():
    return render_template("home.html")

@app.route("/submit_question", methods=["POST"])
def get_quest_results():
    print(request.form.get("the_quest"))
    print(request.form.get("the_doc"))
    print(request.form.get("the_mesh"))
    return render_template("home.html")

if __name__ == '__main__':
    app.run(port=5000, debug=True)
