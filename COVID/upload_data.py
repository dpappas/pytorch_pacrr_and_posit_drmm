
import zipfile, json
from pprint import pprint
from tqdm import tqdm

zip_path    = 'C:\\Users\\dvpap\\Downloads\\db_entries.zip'
archive     = zipfile.ZipFile(zip_path, 'r')
jsondata    = archive.read('db_entries.json')
d           = json.loads(jsondata)


for item in tqdm(d):
    pprint(item)

