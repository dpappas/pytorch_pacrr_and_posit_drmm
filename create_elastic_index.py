
from elasticsearch import Elasticsearch

elastic_con = Elasticsearch(['localhost:9200'],verify_certs=True)

index = 'pubmed_abstracts_index_0_1'
map   = "pubmed_abstracts_mapping_0_1"

elastic_con.indices.delete(
    index=index,
    ignore=[400,404]
)

mapping = {
    "mappings":{
        map:{
            "properties":{
                "AbstractText" : {
                    "type"      : "string",
                    "analyzer"  : "standard",
                },
                'ArticleTitle': {
                    "type": "string",
                    "analyzer": "standard",
                },
                'Title': {
                    "type": "string",
                    "analyzer": "standard",
                },
                'ArticleDate':{
                    "type"      : "date",
                    "format"    : "yyy-MM-dd HH:mm:ss||yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||yyyy/MM/dd||dd/MM/yyyy||epoch_millis||EE MMM d HH:mm:ss Z yyyy"
                },

                "filename": {
                    "type": "string",
                    "index": "not_analyzed",
                },
                'date_type': {
                    "type": "string",
                    "index": "not_analyzed",
                },
                'article_ids' : {
                    "type"      : "string",
                    "index"  : "not_analyzed",
                },
                'section_title': {
                    "type"      : "string",
                    "analyzer"  : "standard",
                },
                'journal_title': {
                    "type"      : "string",
                    "analyzer"  : "standard",
                },
                'article_keywords' : {
                    "type"      : "string",
                    "analyzer"  : "standard",
                },
            }
        }
    }
}

print elastic_con.indices.create(index = index, ignore=400, body=mapping)

















