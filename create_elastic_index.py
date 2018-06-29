
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
                    "type": "text",
                    "analyzer": "standard",
                },
                'ArticleTitle': {
                    "type": "text",
                    "analyzer": "standard",
                },
                'Title': {
                    "type": "text",
                    "analyzer": "standard",
                },
                'ArticleDate':{
                    "type"      : "date",
                    "format"    : "yyy-MM-dd HH:mm:ss||yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||yyyy/MM/dd||dd/MM/yyyy||epoch_millis||EE MMM d HH:mm:ss Z yyyy"
                },
                'Keywords' : {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "raw": {
                            "type":  "keyword"
                        }
                    }
                },
                "MeshHeadings": {
                    "properties": {
                        "UI": {
                            "type": "keyword"
                        },
                        "name": {
                            "type"      : "text",
                            "analyzer"  : "standard",
                            "fields": {
                                "raw": {
                                    "type":  "keyword"
                                }
                            }
                        }
                    }
                },
                "Chemicals": {
                    "properties": {
                        "UI": {
                            "type": "keyword"
                        },
                        "RegistryNumber": {
                            "type": "keyword"
                        },
                        "NameOfSubstance": {
                            "type"      : "text",
                            "analyzer"  : "standard",
                            "fields": {
                                "raw": {
                                    "type":  "keyword"
                                }
                            }
                        },
                    }
                },
                "SupplMeshName": {
                    "properties": {
                        "UI"    : { "type": "keyword" },
                        "Type"  : { "type": "keyword" },
                        "name"  : {
                            "type"      : "text",
                            "analyzer"  : "standard",
                            "fields": {
                                "raw": {
                                    "type":  "keyword"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

print elastic_con.indices.create(index = index, ignore=400, body=mapping)

'''
mkdir /media/dpappas/Maxtor/ELK_DATA/
vim ~/ELK/elasticsearch-6.2.4/config/elasticsearch.yml
change path.data to /media/dpappas/Maxtor/ELK_DATA/data
change path.logs to /media/dpappas/Maxtor/ELK_DATA/logs

/home/dpappas/ELK/elasticsearch-6.2.4/bin/elasticsearch &
/home/dpappas/ELK/kibana-6.2.4-linux-x86_64/bin/kibana &

vim /media/dpappas/Maxtor/ELK/elasticsearch-6.3.0/config/elasticsearch.yml
/media/dpappas/Maxtor/ELK/elasticsearch-6.3.0/bin/elasticsearch &
/media/dpappas/Maxtor/ELK/kibana-6.3.0-linux-x86_64/bin/kibana &

'''












