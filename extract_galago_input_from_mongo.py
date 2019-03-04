

import  os, sys, pymongo, dicttoxml
from    pprint import pprint
from    tqdm import tqdm

db_name             = 'pubmedBaseline2019'
collection_name     = 'articles'
client              = pymongo.MongoClient("localhost", 27017, maxPoolSize=50)
mongo_collection    = client[db_name][collection_name]
print(mongo_collection.count())

pprint(mongo_collection.find_one())



'''
<DOC>
<DOCNO>40</DOCNO>
<TEXT>
human brain and placental choline acetyltransferase purification and properties choline acetyltransferase ec 2316 catalyzes the biosynthesis of acetylcholine according to the following chemical equation acetyl-coa choline in equilibrium to acetylcholine coa in addition to nervous tissue primate placenta is the only other animal source which contains appreciable acetylcholine and its biosynthetic enzyme human brain caudate nucleus and human placental choline acetyltransferase were purified to electrophoretic homogeneity using ion-exchange and blue dextran-sepharose affinity chromatography the molecular weights determined by sephadex g-150 gel filtration and sodium dodecyl sulfate gel electrophoresis are 67000 plus or minus 3000 n-ethylmaleimide p-chloromercuribenzoate and dithiobis2-nitrobenzoic acid inhibit the enzyme dithiothreitol reverses the inhibition produced by the latter two reagents the pka of the group associated with n-ethylmaleimide inhibition is 86 plus or minus 03 a chemically competent acetyl-thioenzyme is isolable by sephadex gel filtration the enzymes from the brain and placenta are thus far physically and biochemically indistinguishable
</TEXT>
</DOC>
'''









