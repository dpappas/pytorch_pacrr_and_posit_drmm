
from retrieve_and_rerank import  get_embeds, idf_val, idf, max_idf, wv, prep_data, model, tokenize, np, get_first_n_1

quest = 'A pneumonia outbreak associated with a new coronavirus of probable bat origin'
docs = (
    get_first_n_1(
        qtext       = quest,
        n           = 100,
        max_year    = 2021
    )
)

quest_tokens, quest_embeds = get_embeds(tokenize(quest), wv)
q_idfs = np.array([[idf_val(qw, idf, max_idf)] for qw in quest_tokens], 'float')
for ddd in docs['retrieved_documents']:
    datum = prep_data(quest, ddd['doc'], ddd['norm_bm25_score'], wv, [], idf, max_idf, True)
    doc_emit_, gs_emits_    = model.emit_one(
        doc1_sents_embeds   = datum['sents_embeds'],
        question_embeds     = quest_embeds,
        q_idfs              = q_idfs,
        sents_gaf           = datum['sents_escores'],
        doc_gaf             = datum['doc_af']
    )
    break
