

from tqdm import tqdm
from pprint import pprint
import tarfile, json
targz_path = "/media/dpappas/dpappas_data/CORD_allenai_datasets/2020-11-29/document_parses.tar.gz"
tar = tarfile.open(targz_path, "r:gz")
total_items         = 0
total_paragraphs    = 0
database_instances  = []
for member in tqdm(tar, total=234501):
    # print(member.name)
    total_items += 1
    f = tar.extractfile(member)
    if f is not None:
        c = 0
        d = json.loads(f.read())
        title       = d['metadata']['title']
        database_instances.append({
            'id'    : d['paper_id']+ ' ' +str(c),
            'text'  : title,
            'type'  : 'title',
        })
        c += 1
        if 'abstract' in d:
            abstract    = '\n'.join([t['text'] for t in d['abstract']])
            database_instances.append({
                'id'    : d['paper_id']+ ' ' +str(c),
                'text'  : abstract,
                'type'  : 'abstract',
                'rank'  : c
            })
            c += 1
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
                database_instances.append({
                    'id'    : d['paper_id']+ ' ' +str(c),
                    'text': par_text,
                    'type': 'paragraph',
                    'rank'  : c
                })
                c += 1
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
                database_instances.append({
                    'id'    : d['paper_id']+ ' ' +str(c),
                    'text'  : par_text,
                    'type'  : 'paragraph',
                    'rank'  : c
                })
                c += 1
