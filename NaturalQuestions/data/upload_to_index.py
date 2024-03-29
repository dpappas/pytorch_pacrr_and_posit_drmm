
import gzip, re, json, hashlib, os, random
from tqdm import tqdm
from pprint import pprint
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def clean_soup(html):
    #######################################################
    html    = html.replace('<H1', '\n\n<div><p> \n </p></div> __H__ <div><p> \n </p></div> <H1')
    html    = html.replace('<H2', '\n\n<div><p> \n </p></div> __H__ <div><p> \n </p></div> <H2')
    html    = html.replace('<H3', '\n\n<div><p> \n </p></div> __H__ <div><p> \n </p></div> <H3')
    #######################################################
    html    = html.replace('/H1>', '/H1>\n\n<div><p> \n </p></div>')
    html    = html.replace('/H2>', '/H2>\n\n<div><p> \n </p></div>')
    html    = html.replace('/H3>', '/H3>\n\n<div><p> \n </p></div>')
    #######################################################
    html    = re.sub('<!--.*?-->', '', html, flags=re.DOTALL)
    soupa   = BeautifulSoup(html, 'lxml')
    #######################
    for item in soupa.findAll('style'):
        gb = item.extract()
    #######################
    for item in soupa.findAll('div', {'role': "navigation"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('div', {'class': "suggestions"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('div', {'class': "catlinks"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('div', {'id': "footer"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('', {'class': "references"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('', {'class': "toc"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('', {'class': "mw-editsection"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('', {'role': "note"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('div', {'id': "mw-navigation"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('div', {'class': "mw-jump"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('div', {'class': "thumb"}):
        gb = item.extract()
    #######################
    for item in soupa.findAll('svg'):
        gb = item.extract()
    #######################
    for item in soupa.findAll('table'):
        gb = item.extract()
    #######################
    for item in soupa.findAll('sup', {'class': "reference"}):
        gb = item.extract()
    #######################
    return soupa

def create_an_action(elk_dato, id):
    elk_dato['_id']      = id
    elk_dato['_op_type'] = 'index'
    elk_dato['_index']   = index
    # elk_dato['_type']    = doc_type
    return elk_dato

def upload_to_elk(finished=False):
    global actions
    global b_size
    if(len(actions) >= b_size) or (len(actions)>0 and finished):
        flag = True
        while (flag):
            try:
                result = bulk(es, iter(actions))
                pprint(result)
                flag = False
            except Exception as e:
                if ('ConnectionTimeout' in str(e) or 'rejected execution of' in str(e)):
                    print('Retrying')
                else:
                    print(e)
                    flag = False
        actions         = []

es = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

index       = 'natural_questions_0_1'
# doc_type    = 'natural_questions_map_0_1'
actions     = []
b_size      = 200

# nq_jsonl = '/media/dpappas/dpappas_data/natural_questions/natural_questions/v1.0/sample/nq-dev-sample.jsonl.gz'
# nq_jsonl = '/media/dpappas/dpappas_data/natural_questions/natural_questions/v1.0/sample/nq-train-sample.jsonl.gz'

diri = '/media/dpappas/dpappas_data/NaturalQuestions/natural_questions/v1.0/'

all_fs = []
for subdir in os.listdir(diri):
    subd = os.path.join(diri, subdir)
    for fpath in tqdm(os.listdir(subd)):
        all_fs.append(os.path.join(subd, fpath))

random.shuffle(all_fs)
for nq_jsonl in tqdm(all_fs):
    with open(nq_jsonl, 'rb') as fileobj:
        f = gzip.GzipFile(fileobj=fileobj)
        for l in tqdm(f): # .readlines()):
            dd = json.loads(l.decode('utf-8'))
            ########################
            html = dd['document_html']
            soupa = clean_soup(html)
            ########################
            all_ps = [item.text.strip() for item in soupa.findAll('p') if(len(item.text.strip())>0)]
            ########################
            for i in range(len(all_ps)):
                textt   = '{}:{}'.format(dd['document_title'], i)
                result  = hashlib.md5(textt.encode()).hexdigest()
                jdata   = {
                    '_id'               : result,
                    'paragraph_index'   : i,
                    'total_paragraphs'  : len(all_ps),
                    'paragraph_text'    : all_ps[i],
                    'document_title'    : dd['document_title'],
                    'document_url'      : dd['document_url']
                }
                ######################################
                actions.append(create_an_action(jdata, jdata['_id']))
                upload_to_elk(finished=False)
                ######################################
        fileobj.close()
        ######################################
        upload_to_elk(finished=True)



