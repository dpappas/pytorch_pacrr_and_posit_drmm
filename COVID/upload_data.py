
import zipfile, json
from pprint import pprint

zip_path    = 'C:\\Users\\dvpap\\Downloads\\db_entries.zip'
archive     = zipfile.ZipFile(zip_path, 'r')
jsondata    = archive.read('db_entries.json')
d           = json.loads(jsondata)

pprint(d.keys())