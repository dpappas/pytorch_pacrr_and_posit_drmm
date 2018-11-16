

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





















