
import nltk

sentence    = "Is modified vaccinia Ankara effective for smallpox ?"
grammar     = ('''NP: {<DT>?<VBN>?<JJ>*((<NN>)|(<NNP>))}''')
chunkParser = nltk.RegexpParser(grammar)
tagged      = nltk.pos_tag(nltk.word_tokenize(sentence))
tree        = chunkParser.parse(tagged)


for subtree in list(tree.subtrees())[1:]:
    tttt = subtree.flatten()
    print(' '.join([t[0] for t in tttt]))

