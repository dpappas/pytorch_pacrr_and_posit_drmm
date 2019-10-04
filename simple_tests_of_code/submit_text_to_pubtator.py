
import requests
import time
v
def SubmitText(InputSTR,Bioconcept):
    json = {}
    #
    r = requests.post("https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/submit/"+Bioconcept, data=InputSTR)
    if r.status_code != 200:
        print("[Error]: HTTP code "+ str(r.status_code))
    else:
        SessionNumber = r.text.encode("utf-8")
        print("Thanks for your submission. The session number is : "+ SessionNumber + "\n")
        r       = requests.get("https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/retrieve/" + SessionNumber)
        code    = 404
        while(code != 200):
            time.sleep(5)
            r = requests.get("https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/retrieve/" + SessionNumber)
            code = r.status_code
            response = r.text.encode("utf-8")
            print(code)
        print(response)

SubmitText('Is ibudilast effective for multiple sclerosis?', 'ALL')










