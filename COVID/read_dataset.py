

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
        if('PMC' in member.name):
            total_paragraphs += 1
            total_paragraphs += len(d['body_text'])
        else:
            total_paragraphs += 1
            total_paragraphs += len(d['body_text'])


