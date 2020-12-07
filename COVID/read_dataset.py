

from tqdm import tqdm
from pprint import pprint
import tarfile, json
targz_path = "/media/dpappas/dpappas_data/CORD_allenai_datasets/2020-11-29/document_parses.tar.gz"
tar = tarfile.open(targz_path, "r:gz")
total_items         = 0
total_paragraphs    = 0
for member in tqdm(tar, total=234501):
    # print(member.name)
    total_items += 1
    f = tar.extractfile(member)
    if f is not None:
        d = json.loads(f.read())
        title       = d['metadata']['title']
        if 'abstract' in d:
            abstract    = '\n'.join([t['text'] for t in d['abstract']])
        # lezantes    = '\n'.join([t['text'] for t in d['ref_entries'].values()])
        if('PMC' in member.name):
            for par in d['body_text']:
                lezantes = '\n'.join(
                    [
                        d['ref_entries'][ref_item['ref_id']]['text']
                        for ref_item in par['ref_spans']
                        if ref_item['ref_id']
                    ]
                )
                par_text = par['text'] + '\n\n' + lezantes
                par_text = par_text.strip()
        else:
            for par in d['body_text']:
                lezantes = '\n'.join(
                    [
                        d['ref_entries'][ref_item['ref_id']]['text']
                        for ref_item in par['ref_spans']
                        if ref_item['ref_id']
                    ]
                )
                par_text = par['text'] + '\n\n' + lezantes
                par_text = par_text.strip()

