
from elasticsearch import Elasticsearch

elastic_con = Elasticsearch(['localhost:9200'],verify_certs=True)

index       = 'pubmed_abstracts_index_0_1'
doc_type    = "pubmed_abstracts_mapping_0_1"

elastic_con.indices.delete(
    index=index,
    ignore=[400,404]
)

mapping = {
     "mappings": {
         "page": {
            "properties": {
               "auxiliary_text": {
                  "type": "text"
               },
               "category": {
                  "type": "text"
               },
               "coordinates": {
                  "properties": {
                     "coord": {
                        "properties": {
                           "lat": {
                              "type": "double"
                           },
                           "lon": {
                              "type": "double"
                           }
                        }
                     },
                     "country": {
                        "type": "text"
                     },
                     "dim": {
                        "type": "long"
                     },
                     "globe": {
                        "type": "text"
                     },
                     "name": {
                        "type": "text"
                     },
                     "primary": {
                        "type": "boolean"
                     },
                     "region": {
                        "type": "text"
                     },
                     "type": {
                        "type": "text"
                     }
                  }
               },
               "defaultsort": {
                  "type": "boolean"
               },
               "external_link": {
                  "type": "text"
               },
               "heading": {
                  "type": "text"
               },
               "incoming_links": {
                  "type": "long"
               },
               "language": {
                  "type": "text"
               },
               "namespace": {
                  "type": "long"
               },
               "namespace_text": {
                  "type": "text"
               },
               "opening_text": {
                  "type": "text"
               },
               "outgoing_link": {
                  "type": "text"
               },
               "popularity_score": {
                  "type": "double"
               },
               "redirect": {
                  "properties": {
                     "namespace": {
                        "type": "long"
                     },
                     "title": {
                        "type": "text"
                     }
                  }
               },
               "score": {
                  "type": "double"
               },
               "source_text": {
                  "type": "text"
               },
               "template": {
                  "type": "text"
               },
               "text": {
                  "type": "text"
               },
               "text_bytes": {
                  "type": "long"
               },
               "timestamp": {
                  "type": "date",
                  "format": "strict_date_optional_time||epoch_millis"
               },
               "title": {
                  "type": "text"
               },
               "version": {
                  "type": "long"
               },
               "version_type": {
                  "type": "text"
               },
               "wiki": {
                  "type": "text"
               },
               "wikibase_item": {
                  "type": "text"
               }
            }
         }
    }
}

print elastic_con.indices.create(index = index, ignore=400, body=mapping)
















































