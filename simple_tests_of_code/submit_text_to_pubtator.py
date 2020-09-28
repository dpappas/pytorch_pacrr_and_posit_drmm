
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
            time.sleep(30)
            r = requests.get("https://www.ncbi.nlm.nih.gov/research/pubtator-api/annotations/annotate/retrieve/" + SessionNumber)
            code = r.status_code
            response = r.text #.encode("utf-8")
            print(code, r.text)
        # print(response)
        return response

response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'BioConcept')
print(20*'-')
# response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'gene')
# print(20*'-')
# response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'disease')
# print(20*'-')
# response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'chemical')
# print(20*'-')
# response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'species')
# print(20*'-')
# response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'mutation')
# print(20*'-')
# response = SubmitText('Is ibudilast effective for multiple sclerosis?', 'cellline')
# print(20*'-')


'''
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker

nlp                     = spacy.load("/home/dpappas/en_core_sci_lg-0.2.3/en_core_sci_lg/en_core_sci_lg-0.2.3/")
linker                  = UmlsEntityLinker(resolve_abbreviations=True)
# abbreviation_pipe     = AbbreviationDetector(nlp)

# nlp.add_pipe(abbreviation_pipe)
nlp.add_pipe(linker)

texts = [
    "Is ibudilast effective for multiple sclerosis?",
    'Cemiplimab is used for treatment of which cancer?',
    'What is hemolacria?',
    'What is the purpose of the Ottawa Ankle Rule?',
    'Which enzymes are inhibited by Duvelisib?',
    'What is the mechanism of the drug CRT0066101?',
    'What is known about the gene MIR140?'
]

for text in texts:
    print(text)
    doc = nlp(text)
    for entity in doc.ents:
        print(20 *'-')
        print("Name: ", entity)
        for umls_ent in entity._.umls_ents:
            print(linker.umls.cui_to_entity[umls_ent[0]])
    print(20 *'#')

'''







