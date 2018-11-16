
from elasticsearch import Elasticsearch

elastic_con = Elasticsearch(['localhost:9200'], verify_certs=True)
index       = 'wikipedia_json_gz'
doc_type    = "wiki_page"

elastic_con.indices.delete(index=index, ignore=[400,404])

mapping = {
     "mappings": {
         doc_type: {
            "properties": {
               "auxiliary_text": {
                  "type": "text"
               },
               'category': {
                  "type": "text",
                  "fields": {
                     "raw": {
                        "type": "keyword"
                     }
                  }
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
               # "defaultsort": {
               #    "type": "boolean"
               # },
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
               'title': {
                  "type": "text",
                  "fields": {
                     "raw": {
                        "type": "keyword"
                     }
                  }
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

# zcat enwiki-20181112-cirrussearch-content.json.gz | parallel --pipe -L 2 -N 200 -j3 'curl -s http://localhost:9200/wikipedia_json_gz/_bulk --data-binary @- > /dev/null'













































