import zipfile
import json
from pprint import pprint
from tqdm import tqdm

archive = zipfile.ZipFile('/home/dpappas/Downloads/BioASQ-training7b.zip', 'r')

jsondata = archive.read('BioASQ-training7b/trainining7b.json')

d = json.loads(jsondata)

maxx = 0
for q in tqdm(d['questions']):
    for link in q['documents']:
        t = int(link.split('/')[-1])
        maxx = max([maxx, t])

print('https://www.ncbi.nlm.nih.gov/pubmed/{}'.format(maxx))

print(q['body'])
print(list(t.split('/')[-1] for t in q['documents']))

'''


GET pubmed_abstracts_0_1/_search
{
  "_source": ["ArticleTitle", "pmid"],
  "size" : 100,
  "query": {
    "bool": {
      "should": [
        {
          "multi_match" : {
            "query"                 : "What is Mendelian randomization",
            "type"                  : "best_fields",
            "fields"                : ["ArticleTitle", "AbstractText"],
            "minimum_should_match"  : "50%",
            "slop"                  : 2
          }
        }
      ]
    }
  }
}

'''
