#!/usr/bin/env python
# -*- coding: utf-8 -*-

# zcat enwiki-20181112-cirrussearch-content.json.gz | head -5000 | tail -1 | python -m json.tool
# https://stackoverflow.com/questions/47476122/loading-wikipedia-dump-into-elasticsearch

# bzcat enwiki-latest-pages-articles.xml.bz2 | head -100

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import subprocess
from bs4 import BeautifulSoup
from pprint import pprint
import re
from lxml import etree as et
fpath   = '/media/dpappas/dpappas_data/enwiki-latest-pages-articles.xml.bz2'
proc    = subprocess.Popen(["bzcat",fpath], stdout=subprocess.PIPE)
temp_text = ''
while True:
    line = proc.stdout.readline().rstrip()
    if('<page>' in line):
        temp_text   = ''
        temp_text   += '\n'+line
        print "test:", line
    elif ('</page>' in line):
        temp_text   += '\n'+line
        annots      = re.findall(r'\{\{.?\}\}',      temp_text, flags=re.DOTALL)
        temp_text   = re.sub(r'\{\{.?\}\}', '', temp_text, flags=re.DOTALL)
        # temp_text = temp_text.replace('&gt;', '>')
        # temp_text = temp_text.replace('&lt;', '<')
        soup = BeautifulSoup(temp_text, 'lxml')
        print soup.prettify()
        pprint(annots)
        print 20 * '='
        temp_text = ''
    else:
        temp_text += '\n'+line





exit()

import subprocess
from lxml import etree as et
from bz2file import BZ2File
fpath   = '/media/dpappas/dpappas_data/enwiki-latest-pages-articles.xml.bz2'
p       = subprocess.Popen(["bzcat",fpath], stdout=subprocess.PIPE)
parser = et.iterparse(p.stdout, tag='page')

for events, elem in parser:
    if('page' in elem.tag.lower()):
        print(elem.tag)
        print(et.tostring(elem))
    elem.clear()


exit()

import libarchive.public

fpath   = '/media/dpappas/dpappas_data/wikipedia-en-html.tar.7z'
with libarchive.public.file_reader(fpath) as e:
    for entry in e:
        for block in entry.get_blocks():
            print(block)

exit()

fpath   = '/media/dpappas/dpappas_data/enwiki-latest-pages-articles.xml.bz2'
infile  = gzip.open(fpath)
content = infile.read()
children = etree.fromstring(content).getchildren()
ch_counter = 0
for ch_tree in children:
    ch_counter += 1
    for elem in ch_tree.iter(tag='MedlineCitation'):
        # elem = etree.fromstring(etree.tostring(elem))
        print(etree.tostring(elem))
        break
    break

exit()

# wget https://dumps.wikimedia.org/other/static_html_dumps/current/en/wikipedia-en-html.tar.7z
#
# 7z x -so yourfile.tar.7z | tar xf - -C target_dir
# 7z x -so wikipedia-en-html.tar.7z | tar xf - -C wikipedia-en-htmls
#

from bs4 import BeautifulSoup
from pprint import pprint
import tarfile

filename    = '/media/dpappas/dpappas_data/wikipedia-en-html.tar'
tar         = tarfile.open(filename)
nonos       = ['Image', 'User', 'Talk', 'Category']
for member_info in tar:
    if(not any([nono in member_info.name for nono in nonos])):
        print member_info.name
        f       = tar.extractfile(member_info)
        content = f.read()
        soup    = BeautifulSoup(content, "lxml")
        # print   soup.prettify()
        try:
            pprint(soup.find('div', {'id': 'bodyContent'}).find_all('div', {'id': 'mw-content-text'}))
        except:
            None
    # break

f       = tar.extractfile('en/articles//H/i/n/Hinge_loss')
content = f.read()
soup    = BeautifulSoup(content, "lxml")





















