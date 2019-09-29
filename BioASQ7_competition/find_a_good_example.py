
from collections import Counter
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

batch = 1
fg = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/BioASQ-task7bPhaseB-testset{}'.format(batch, batch)
f1 = '/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_{}/jpdrmm.json'.format(batch)
f2 = '/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_{}/pdrmm_pdrmm.json'.format(batch)

gt = json.load(open(fg))
gt = dict((item['id'], item) for item in gt['questions'])
s1 = json.load(open(f1))
s1 = dict((item['id'], item) for item in s1['questions'])
s2 = json.load(open(f2))
s2 = dict((item['id'], item) for item in s2['questions'])

len_max = 30
held_out = {}
for k in s1:
    gt_snips = []
    for item in gt[k]['snippets']:
        gt_snips.extend(list(zip(sent_tokenize(item['text'].strip()), len(sent_tokenize(item['text'].strip()))*[item['document']])))
    # The annotators should not highlight the entire document
    gold_counter    = Counter([sn[1] for sn in gt_snips])
    # Also the models must have seen the same docs to decide
    common_docs     = set(s1[k]['documents']).intersection(s2[k]['documents'])
    ########################################
    clean_gold      = [' '.join(bioclean(sn['text'])) for sn in gt[k]['snippets']]
    qtext    = gt[k]['body']
    s1_snips = [(sn['text'], sn['document']) for sn in s1[k]['snippets'] if(sn['document'] in gt[k]['documents'] and gold_counter[sn['document']]<4)]
    s2_snips = [(sn['text'], sn['document']) for sn in s2[k]['snippets'] if(sn['document'] in gt[k]['documents'] and gold_counter[sn['document']]<4)]
    ########################################
    found_by_1, found_by_2 = [], []
    for snip in s1_snips:
        if(snip_is_relevant(' '.join(bioclean(snip[0])), clean_gold)):
            if(snip[1] in common_docs):
                if(len(bioclean(snip[0]))<len_max):
                    found_by_1.append(snip)
    for snip in s2_snips:
        if (snip_is_relevant(' '.join(bioclean(snip[0])), clean_gold)):
            if(snip[1] in common_docs):
                if(len(bioclean(snip[0]))<len_max):
                    found_by_2.append(snip)
    ########################################
    # common_docs = set([t[1] for t in found_by_1]).intersection([t[1] for t in found_by_2])
    # found_by_1  = [t for t in found_by_1 if(t[1] in common_docs)]
    # found_by_2  = [t for t in found_by_2 if(t[1] in common_docs)]
    common_snips    = set(found_by_1).intersection(found_by_2)
    found_by_1      = set(found_by_1) - common_snips
    found_by_2      = set(found_by_2) - common_snips
    if(len(found_by_1)!=0 and len(found_by_2)!=0):
        held_out[qtext] = (gt_snips, found_by_1, found_by_2)

print(len(held_out))
pprint(list(held_out.keys()))

pprint(held_out['Is ibudilast effective for multiple sclerosis?'])
pprint(held_out['Cemiplimab is used for treatment of which cancer?'])
pprint(held_out['What is hemolacria?'])
pprint(held_out['What is the purpose of the Ottawa Ankle Rule?'])
# pprint(held_out['Which enzymes are inhibited by Duvelisib?'])
pprint(held_out['What is the mechanism of the drug CRT0066101?'])
pprint(held_out['What is known about the gene MIR140?'])

'''
for k in s1:
    s1_snips = [sn['text'] for sn in s1[k]['snippets']]
    s2_snips = [sn['text'] for sn in s2[k]['snippets']]
    # gt_snips = [sn['text'] for sn in gt[k]['snippets']]
    gt_snips = []
    for item in gt[k]['snippets']:
        gt_snips.extend(sent_tokenize(item['text'].strip()))
    clean_gold = [' '.join(bioclean(sn['text'])) for sn in gt[k]['snippets']]
    if(len(set(s1_snips)-set(s2_snips))>6):
        rel_snips_1 = 0
        for snip in s1_snips:
            if(snip_is_relevant(' '.join(bioclean(snip)), clean_gold)):
                rel_snips_1 += 1
        rel_snips_3 = 0
        for snip in s2_snips:
            if (snip_is_relevant(' '.join(bioclean(snip)), clean_gold)):
                rel_snips_3 += 1
        if(rel_snips_1-rel_snips_3>2 and rel_snips_1<len(clean_gold)):
            print(s1[k]['body'])
            print(20 * '-')
            for s in gt_snips:
                print(s)
            print(20 * '-')
            for s in s1_snips:
                print(s)
            print(20 * '-')
            for s in s2_snips:
                print(s)
            print(20 * '=')
'''





