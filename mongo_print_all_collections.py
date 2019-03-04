import pymongo, json
from pprint import pprint

client  = pymongo.MongoClient("localhost", 27017, maxPoolSize=50)
d       = dict(
    (
        db,
        [
            [
                collection,
                client[db][collection].count(),
                # client[db][collection].find_one()
            ]
            for collection in client[db].collection_names()
        ]
    )
    for db in client.database_names()
)
pprint(d)

pprint(client['pubmedBaseline2018']['articles'].find_one())

