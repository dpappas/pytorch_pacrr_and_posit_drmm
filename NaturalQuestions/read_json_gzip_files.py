
import gzip, re, json, hashlib
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

# nq_jsonl = '/media/dpappas/dpappas_data/natural_questions/natural_questions/v1.0/sample/nq-dev-sample.jsonl.gz'
nq_jsonl = '/media/dpappas/dpappas_data/natural_questions/natural_questions/v1.0/sample/nq-train-sample.jsonl.gz'
with open(nq_jsonl, 'rb') as fileobj:
    f = gzip.GzipFile(fileobj=fileobj)
    for l in f.readlines():
        dd = json.loads(l.decode('utf-8'))
        # # pprint(dd)
        # print(20 * '-')
        # print(dd['example_id'])
        # print(10 * '-')
        # print(dd['question_text'])
        # print(20 * '-')
        # print(dd['question_tokens'])
        # print(10 * '-')
        # print(dd['document_title'])
        # print(20 * '-')
        # print(dd['document_url'])
        # print(20 * '-')
        # pprint(dd['document_tokens'])
        # print(10 * '-')
        # pprint(dd['annotations'])
        # print(20 * '-')
        # pprint(dd['long_answer_candidates'])
        # print(20 * '-')
        # break
        ########################
        # pprint(dd.keys())
        ########################
        html = dd['document_html']
        soupa = clean_soup(html)
        ########################
        # print(soupa.find('body').text.strip())
        # print(soupa.prettify())
        ########################
        all_ps = [item.text.strip() for item in soupa.findAll('p') if(len(item.text.strip())>0)]
        # print('\n\n'.join(all_ps))
        ########################
        for i in range(len(all_ps)):
            textt   = '{}:{}'.format(dd['document_title'], i)
            result  = hashlib.md5(textt.encode()).hexdigest()
            datum   = {
                '_id'               : result,
                'paragraph_index'   : i,
                'paragraph_text'    : all_ps[i],
                'document_title'    : dd['document_title'],
                'document_url'      : dd['document_url']
            }
            pprint(datum)
        ########################
        # # pprint(dd['annotations'])
        # print(10 * '-')
        # print('Q: ' + dd['question_text'])
        # for annot in dd['annotations']:
        #     if(annot['long_answer']['candidate_index'] != -1):
        #         s = annot['long_answer']['start_token']
        #         e = annot['long_answer']['end_token']
        #         print('L: '+' '.join([t['token'] for t in dd['document_tokens'][s:e+1]]))
        #     for sa in annot['short_answers']:
        #         s = sa['start_token']
        #         e = sa['end_token']
        #         print('S: '+' '.join([t['token'] for t in dd['document_tokens'][s:e+1]]))
        # ########################
        # print(20 * '-')
    fileobj.close()

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




