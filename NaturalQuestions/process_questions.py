
import gzip, re, json, hashlib, os, random
from tqdm import tqdm
from pprint import pprint
from bs4 import BeautifulSoup

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

diri = '/media/dpappas/dpappas_data/natural_questions/natural_questions/v1.0/'

all_fs = []
for subdir in  os.listdir(diri):
    subd = os.path.join(diri, subdir)
    for fpath in tqdm(os.listdir(subd)):
        all_fs.append(os.path.join(subd, fpath))

random.shuffle(all_fs)

all_questions = []
for nq_jsonl in tqdm(all_fs):
    with open(nq_jsonl, 'rb') as fileobj:
        f = gzip.GzipFile(fileobj=fileobj)
        for l in tqdm(f.readlines()):
            dd = json.loads(l.decode('utf-8'))
            ########################
            html = dd['document_html']
            soupa = clean_soup(html)
            ########################
            all_ps = [item.text.strip() for item in soupa.findAll('p') if(len(item.text.strip())>0)]
            ########################
            for annot in dd['annotations']:
                if (annot['long_answer']['candidate_index'] != -1):
                    s = annot['long_answer']['start_token']
                    e = annot['long_answer']['end_token']
                    # print(os.path.basename(os.path.abspath(os.path.join(nq_jsonl, os.pardir))))
                    # print('Title: ' + dd['document_title'])
                    # print('Link: ' + dd['document_url'])
                    # print('Q: ' + dd['question_text'])
                    # print('L: ' + )
                    for sa in annot['short_answers']:
                        s = sa['start_token']
                        e = sa['end_token']
                        # print('S: ' + ' '.join([t['token'] for t in dd['document_tokens'][s:e + 1]]))
                        print(10 * '-')
                        datum = {
                            'example_id'     : dd['example_id'],
                            'dataset'        : os.path.basename(os.path.abspath(os.path.join(nq_jsonl, os.pardir))),
                            'document_title' : dd['document_title'],
                            'document_url'   : dd['document_url'],
                            'question'       : dd['question_text'],
                            'long_answer'    : ' '.join([t['token'] for t in dd['document_tokens'][s:e + 1]]),
                            'short_answer'   : ' '.join([t['token'] for t in dd['document_tokens'][s:e + 1]])
                        }
                        # pprint(datum)
                        all_questions.append(datum)
            ########################
            fileobj.close()

train_questions = [q for q in  all_questions if(q['dataset'] == 'train')]
dev_questions   = [q for q in  all_questions if(q['dataset'] == 'dev')]

print(len(all_questions))
print(len(train_questions))
print(len(dev_questions))

'''

# 'annotations', 
# 'document_html', 
# 'document_title', 
'document_tokens', 
# 'document_url', 
# 'example_id', 
# 'long_answer_candidates', 
# 'question_text', 
# 'question_tokens'

'''


