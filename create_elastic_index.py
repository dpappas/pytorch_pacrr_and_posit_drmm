
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

















