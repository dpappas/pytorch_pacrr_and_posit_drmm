

# wget https://dumps.wikimedia.org/other/static_html_dumps/current/en/wikipedia-en-html.tar.7z
#
# 7z x -so yourfile.tar.7z | tar xf - -C target_dir
# 7z x -so wikipedia-en-html.tar.7z | tar xf - -C wikipedia-en-htmls
#

from bs4 import BeautifulSoup
import tarfile

filename    = '/media/dpappas/dpappas_data/wikipedia-en-html.tar'
tar         = tarfile.open(filename)
for member_info in tar:
    print member_info.name
    f       = tar.extractfile(member_info)
    content = f.read()
    soup    = BeautifulSoup(content, "lxml")
    print   soup.prettify()
    break






















