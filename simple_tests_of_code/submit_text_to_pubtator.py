
import requests
import time

def SubmitText(InputSTR,Bioconcept):
    json = {}
    #
    r = requests.post("https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/submit/"+Bioconcept, data=InputSTR)
    if r.status_code != 200:
        print("[Error]: HTTP code "+ str(r.status_code))
        return None
    else:
        SessionNumber = r.text #.encode("utf-8")
        print("Thanks for your submission. The session number is : "+ SessionNumber + "\n")
        code     = 404
        response = None
        while(code != 200):
            time.sleep(5)
            r = requests.get("https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/retrieve/" + SessionNumber)
            code = r.status_code
            response = r.text #.encode("utf-8")
            print(code)
        print(response)
        return response

response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'gene')
print(20*'-')
response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'disease')
print(20*'-')
response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'chemical')
print(20*'-')
response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'species')
print(20*'-')
response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'mutation')
print(20*'-')
response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'cellline')
print(20*'-')










