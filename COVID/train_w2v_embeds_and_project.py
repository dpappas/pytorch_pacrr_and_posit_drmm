
from    tqdm            import tqdm
from    pprint          import pprint
import  gensim, os
import  logging
import re
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

bioclean    = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

index       = 'covid_index_0_1'
elastic_con = Elasticsearch(['127.0.01:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

def yield_lines(elastic_con, index):
    items = scan(elastic_con, query=None, index=index, request_timeout=3000)
    total = elastic_con.count(index=index)['count']
    for item in tqdm(items, total=total):
        text = item['_source']['joint_text']
        title       = text.split(30*'-')[0].strip()
        yield bioclean(title)
        abstract    = text.split(30*'-')[1].strip()
        yield bioclean(abstract)

class MakeIter(object):
    def __init__(self, elastic_con, index):
        self.elastic_con    = elastic_con
        self.index          = index
    def __iter__(self):
        for item in yield_lines(self.elastic_con, self.index):
            yield item

iterator1   = MakeIter(elastic_con, index)
# pprint(iterator1.__iter__().__next__())
total_examples = 1121501 * 2

################################################################################################

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
size        = 30
model       = gensim.models.Word2Vec(iterator1, size=size, window=5, min_count=4, workers=10)
model.train(iterator1, total_examples=total_examples, epochs=20)
model.save("covid_19_w2v_embeds_{}.model".format(size))

################################################################################################


