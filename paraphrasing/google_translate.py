
# pip install googletrans

import googletrans
from googletrans import Translator
print(googletrans.LANGUAGES)
translator  = Translator()

question    = "Is modified vaccinia Ankara effective for smallpox?"
print(question)

def get_different(question, lang='fr'):
    result      = translator.translate(question, src='en', dest=lang)
    transl_quest = result.text
    result      = translator.translate(transl_quest, src=lang, dest='en')
    return result.text

for  lang in googletrans.LANGUAGES.keys():
    trans = get_different(question, lang=lang)
    if(trans != question):
        print(lang)
        print(trans)







