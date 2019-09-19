
import  json, re
from    pprint          import pprint
from    nltk.tokenize   import sent_tokenize
bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

def snip_is_relevant(one_sent, gold_snips):
    return int(
        any(
            [
                (one_sent.encode('ascii', 'ignore') in gold_snip.encode('ascii', 'ignore'))
                or
                (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii', 'ignore'))
                for gold_snip in gold_snips
            ]
        )
    )

gt = json.load(open("C:\\Users\\dvpap\\OneDrive\\Desktop\\test_batch_3\\BioASQ-task7bPhaseB-testset3"))
gt = dict((item['id'], item) for item in gt['questions'])
s1 = json.load(open("C:\\Users\\dvpap\\OneDrive\\Desktop\\test_batch_3\\w2v-jpdrmm_system-1.json"))
s1 = dict((item['id'], item) for item in s1['questions'])
s3 = json.load(open("C:\\Users\\dvpap\\OneDrive\\Desktop\\test_batch_3\\rerank-term-pacrr-bcnn_system-3.json"))
s3 = dict((item['id'], item) for item in s3['questions'])

for k in s1:
    s1_snips = [sn['text'] for sn in s1[k]['snippets']]
    s3_snips = [sn['text'] for sn in s3[k]['snippets']]
    # gt_snips = [sn['text'] for sn in gt[k]['snippets']]
    gt_snips = []
    for item in gt[k]['snippets']:
        gt_snips.extend(sent_tokenize(item['text'].strip()))
    clean_gold = [' '.join(bioclean(sn['text'])) for sn in gt[k]['snippets']]
    if(len(set(s1_snips)-set(s3_snips))>6):
        rel_snips_1 = 0
        for snip in s1_snips:
            if(snip_is_relevant(' '.join(bioclean(snip)), clean_gold)):
                rel_snips_1 += 1
        rel_snips_3 = 0
        for snip in s3_snips:
            if (snip_is_relevant(' '.join(bioclean(snip)), clean_gold)):
                rel_snips_3 += 1
        # if(rel_snips_1-rel_snips_3>5):
        # if(rel_snips_3-rel_snips_1>1):
        if(rel_snips_1-rel_snips_3>2 and rel_snips_1<len(clean_gold)):
            print(s1[k]['body'])
            print(20 * '-')
            for s in gt_snips:
                print(s)
            print(20 * '-')
            for s in s1_snips:
                print(s)
            print(20 * '-')
            for s in s3_snips:
                print(s)
            print(20 * '=')

