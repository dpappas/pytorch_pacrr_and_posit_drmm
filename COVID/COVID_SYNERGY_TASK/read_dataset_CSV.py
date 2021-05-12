
from tqdm import tqdm
from pprint import pprint
import tarfile, json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import csv, os
from csv import reader

def create_an_action(elk_dato, id):
    elk_dato['_id']      = id
    elk_dato['_op_type'] = 'index'
    elk_dato['_index']   = index
    return elk_dato

def upload_to_elk():
    global actions
    flag = True
    while (flag):
        try:
            result = bulk(es, iter(actions))
            pprint(result)
            flag = False
        except Exception as e:
            if ('ConnectionTimeout' in str(e) or 'rejected execution of' in str(e)):
                print('Retrying')
            else:
                print(e)
                flag = False
    actions         = []

# index       = 'allenai_covid_index_2020_11_29_csv'
# index       = 'allenai_covid_index_2021_01_10_csv'
index       = 'allenai_covid_index_2021_01_25_csv'
es          = Elasticsearch(['127.0.0.1:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

b_size              = 250
total_items         = 0
total_paragraphs    = 0
database_instances  = []
actions             = []
# csv_path            = '/media/dpappas/dpappas_data/CORD_allenai_datasets/2020-11-29/metadata.csv'
# csv_path            = '/media/dpappas/dpappas_data/CORD_allenai_datasets/2021-01-10/metadata.csv'
csv_path            = '/media/dpappas/dpappas_data/CORD_allenai_datasets/2021-01-25/metadata.csv'
with open(csv_path, 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in tqdm(csv_reader, total=500000):
        (
            cord_uid,
            sha,
            source_x,
            title,
            doi,
            pmcid,
            pubmed_id,
            license,
            abstract,
            publish_time,
            authors,
            journal,
            mag_id,
            who_covidence_id,
            arxiv_id,
            pdf_json_files,
            pmc_json_files,
            url,
            s2_id
        ) = row
        if(len(publish_time)==0):
            publish_time = '1600'
        datum       = {
            '_id'           : cord_uid,
            'joint_text'    : title + '--------------------' + abstract,
            'cord_uid'      : cord_uid,
            'doi'           : doi,
            'pubmed_id'     : pubmed_id,
            'url'           : url,
            'publish_time'  : publish_time
        }
        actions.append(create_an_action(datum, datum['_id']))
        if(len(actions) == b_size):
            upload_to_elk()

upload_to_elk()

# go to
# https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html

