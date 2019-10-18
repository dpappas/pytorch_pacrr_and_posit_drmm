
from .emit_given_text import get_results_for_one_question

from colour import Color
from flask import url_for
from flask import Flask
from flask import render_template
from flask import request

white           = Color("white")
yellow_colors   = list(white.range_to(Color("yellow"), 101))
yellow_colors   = [c.get_hex_l() for c in yellow_colors]
blue_colors     = list(white.range_to(Color("blue"), 101))
blue_colors     = [c.get_hex_l() for c in blue_colors]

app = Flask(__name__)

@app.route("/")
def get_news():
    return render_template("sentence_similarity.html")

# @app.route("/test_similarity_matrix", methods=["POST", "GET"])
# def test_similarity_matrix():
#     sent1           = request.form.get("sent1").strip()
#     sent2           = request.form.get("sent2").strip()
#     tokens1         = tokenize(sent1)
#     tokens2         = tokenize(sent2)
#     tokens1, emb1   = get_embeds(tokens1, wv)
#     tokens2, emb2   = get_embeds(tokens2, wv)
#     scores          = cosine_similarity(emb1, emb2).clip(min=0) * 100
#     # scores          = cosine_similarity(emb1, emb2)
#     # scores          = (scores + 1.0) / 2.0 # max min normalization
#     # scores          = scores * 100
#     _, _, scores_2  = create_one_hot_and_sim(tokens1, tokens2)
#     #############
#     ret_html    = '''
#     <html>
#     <head>
#     <style>
#     table, th, td {border: 1px solid black;}
#     .floatLeft { width: 50%; float: left; }
#     .floatRight {width: 50%; float: right; }
#     .container { overflow: hidden; }
#     </style>
#     </head>
#     <body>
#     '''
#     ret_html    += '<div class="container">'
#     ret_html    += '<div class="floatLeft">'
#     ret_html    += '<p><h2>W2V cosine similarity (clipped negative to zero):</h2></p>'
#     ret_html    += create_table(tokens1, tokens2, scores)
#     ret_html    += '</div>'
#     ret_html    += '<div class="floatRight">'
#     ret_html    += '<p><h2>One-Hot cosine similarity:</h2></p>'
#     ret_html    += create_table(tokens1, tokens2, scores_2*100)
#     ret_html    += '</div>'
#     ret_html    += '</div>'
#     ret_html    += '''
#     <p><b>Note:</b> Attention scores between the sentences:</p>
#     <p>Sentence1: {}</p>
#     <p>Sentence2: {}</p>
#     </body>
#     </html>
#     '''.format(sent1, sent2)
#     return ret_html

if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


'''
batch 2
['Which two drugs are included in the MAVYRET pill?', 'Which transcription factor binding site is contained in Alu repeats?', 'Describe the 4D-CHAINS algorithm', 'What is eravacycline?', 'As of September 2018, what machine learning algorithm is used to for cardiac arrhythmia detection from a  short single-lead ECG recorded by a wearable device?', 'What is latex bead phagocytosis?', 'Is inositol effective for trichotillomania?', 'What is the function of the Nup153 protein?', 'Which de novo truncating mutations in WASF1 cause intellectual disability?', 'Does lucatumumab bind to CD140?', 'Please list symptoms of measles.', 'What is an organoid?', 'Is pazopanib an effective treatment of glioblastoma?', 'What is CIBERSORT used for?', 'Does Groucho related gene 5 (GRG5) have a role only in late development?', 'Can enasidenib be used for the treatment of acute myeloid leukemia?', 'What membrane proteins constitute TAM family of receptor tyrosine kinases (RTKs)?', 'Is collagen the most abundant human protein?', 'Describe symptoms of the Visual snow syndrome.', 'What is the aim of the METABRIC project?', 'Which microRNA is the mediator of the obesity phenotype of patients carrying 1p21.3 microdeletions?', 'Name 4 side effects of enasidenib', 'What disease is treated with Laparoscopic Heller Myotomy (LHM)?', 'Which human disease is experimental autoimmune encephalomyelitis (EAE) model for?', 'Should dacomitinib be used for treatment of glioblastoma patients?', 'What is GeneWeaver used for?', 'Is there any role for HUWE1 in MYC signalling?', 'What is PEGylation?', 'Does epidural anesthesia for pain management during labor affect the Apgar score of the the infant?', 'Is phospholipid hydroperoxide glutathione peroxidase a selenoprotein?', 'What is evaluated with the SAD PERSONS scale?', 'How is the Regulatory Trait Concordance (RTC) calculated?', 'Is TIAM1 favoring tumor progression in colorectal cancer (CRC)?', 'How can PEGylation improve recombinant drugs?', 'Has Glyceraldehyde 3-phosphate dehydrogenase (GAPDH) been reported to be a plasminogen receptor in pathogenic bacteria?', 'What is the function of Plasminogen activator inhibitor 1?', 'Which drugs are included in the Orkambi pill?', 'Which proteins form the nuclear pore basket in human cells?', 'What are the effects of STEF depletion?', 'Which disease is gemtuzumab ozogamicin used for?', 'Can exposure to heavy metals like lead(Pb) or cadmium(Cd) cause changes in DNA methylation patterns in Isoetes sinensis?', 'What is the cause if the rare disease cystinosis?', 'Does simvastatin improve outcomes of aneurysmal subarachnoid hemorrhage?', "Which databases can exchange data using Matchmaker Exchange's API?", 'Where does gemtuzumab ozogamicin bind?', 'Do raspberries improve postprandial glucose and acute and chronic inflammation in adults with type 2 Diabetes?', 'Are Mesenchymal stem cells (MSC) multipotent cells?', 'What is the purpose of the Ottawa Ankle Rule?', 'Are de novo mutations in regulatory elements responsible for neurodevelopmental disorders?', 'Which molecular does daratumumab target?', 'What is PANDAS disease?', 'Are phagosomal proteins ubiquitinated?', 'Does tremelimumab improve survival of mesothelioma patients?', 'What are the effects of the deletion of all three Pcdh clusters (tricluster deletion) in mice?', 'What is CPX351?', 'What micro-RNAs are useful in the diagnosis and prognosis of Heart Failure?', 'List proteins interacting with Star-PAP', 'Which molecule is targeted by Olaratumab?', 'What is the predicted function for TMEM132 family?', 'Can CPX-351 be used for the treatment of tuberculosis?', 'What is etarfolatide used for?', 'What is the role of the Leucosporidium ice-binding protein', 'Does Panitumumab prolong survival of biliary tract cancer patients?', 'Which tool has been developed for GPU-accelerated alignment of bisulfite-treated DNA sequences?', 'What is ivosidenib?', 'A herd immunity of what percentage of the population is required to prevent sporadic outbreaks?', 'What is the normal function p53?', 'Which two drugs are included in the Entresto pill?', 'Is there a deep-learning algorithm for protein solubility prediction?', 'Name 3 diseases for which lucatumumab is being tested?', 'Have machine learning methods been used to predict the severity of major depressive disorder(MDD)?', 'What is ferroptosis?', 'Which disease can be classified with the Awaji Criteria?', 'Which database has been developed that contains experimentally-confirmed carbonylated proteins?', 'Is lucatumumab a polyclonal antibody?', 'Briefly describe a deep learning system that is more accurate than human experts at detecting melanoma.', 'What is known about the gut bacteria and depression.', 'Is lithium effective for treatment of amyotrophic lateral sclerosis?', 'List available R packages for processing NanoString data', 'Can midostaurin inhibit angiogenesis?', 'What is the 3D tomography imaging technique for diagnosis of  eye disease?', 'Is the enzyme ERAP2 associated with the disease birdshot chorioretinopathy?', 'What is the mechanism of action of durvalumab?', 'Are there ultraconserved regions in the budding yeast (Saccharomyces cerevisiae)?', 'For which indications has midostaurin received FDA and EMA approval?', 'Endolymphatic hydrops is associated with Meniere’s disease. Please provide a summary of endoymphatic hydrops including the symptoms and affected body part.', 'List MHC-I-associated inflammatory disorders.', 'What is the mechanism of action of Alpelisib?', 'What is the role of ZCCHC17?', 'Which was the first mutant IDH2 inhibitor to be approved for patients with acute myeloid leukemia?', "What part of the body is affected by Meniere's disease?", 'Is ADP-ribosylation a PTM?', 'What is the mechanism of action of arimoclomol?', 'Describe CGmapTools', 'Which disease is PGT121 used for?', 'Where, in what US state, was there a measles outbreak in an Amish community', 'What is filipin staining used for?', 'Is pacritinib effective for treatment of myelofibrosis?', 'Describe GeneCodeq', 'What type of topoisomerase inhibitor is gepotidacin?']

batch 3
['What is the mechanism of action of anlotinib?', 'Describe Canvas SPW', 'When did delafloxacin receive its first approval in the USA for acute bacterial skin and skin structure infections?', 'What is a prolactinoma and where in the body would they be found?', 'List phagosomal markers.', 'What is the mechanism of action of motolimod?', 'Are there graph kernel libraries available implemented in JAVA?', 'List bacterial families for which delafloxacin has been shown to be effective.', 'What is the association of epigallocatechin with the cardiovascular system?', 'What is the cause of Sandhoff disease?', 'What is the mechanism of action of tucatinib?', 'Are there tools for reviewing variant calls?', 'Name one CCR4 targeted drug.', "What periodontal disease associated bacteria is also associated with Alzheimer's disease?", 'How does the Cholera toxin enter a cell?', 'What is the mechanism of action of cemiplimab?', 'Which algorithm has been developed for finding conserved non-coding elements (CNEs) in genomes?', 'Can mogamulizumab be used for the treatment of cutaneous T-cell lymphoma?', 'What causes Bathing suit Ichthyosis(BSI)?', 'List search engines used in proteomics.', 'Is cabozantinib effective for Hepatocellular Carcinoma?', 'Describe SLIC-CAGE', 'Has ivosidenib been FDA approved for use against acute myeloid leukemia?', 'What is achalasia?', 'When is serum AFP used as marker?', 'Is dupilumab effective for treatment of asthma?', 'Are Copy Number Variants (CNVs) depleted in regions of low mappability?', 'How does botulism toxin act on the muscle?', 'What is the function of the protein encoded by the gene STING?', 'List major features of TEMPI Syndrome.', 'Which random forest method has been developed for detecting Copy Numbers Variants (CNVs)?', 'What is nyctinasty in plants?', 'List STING agonists.', 'Burosumab is used for treatment of which disease?', 'Which deep learning algorithm has been developed for variant calling?', 'What 2 biological processes are regulated by STAMP2 in adipocytes?', 'Cerliponase alfa is apprived for treatment of which disease?', 'What is the percentage of individuals at risk of dominant medically actionable disease?', 'What organism causes hepatic capillariasis?', 'Is verubecestat effective for Alzheimer’s Disease?', 'Can mitochondria be inherited by both parents in humans?', 'In clinical trials, the H3 R antagonist CEP-26401 has a positive effect on cognition, yes or no?', 'Is baricitinib effective for rheumatoid arthritis?', 'What type of data does the UK biobank resource contain?', 'Have yeast prions become important models for the study of the basic mechanisms underlying human amyloid diseases?', 'Describe the mechanism of action of apalutamide.', 'List potential reasons regarding why potentially important genes are ignored', 'What is the indication for Truvada?', 'List drugs that were tested in the CheckMate 214 trial.', 'What is the contribution of ultraconserved elements in Australasian smurf-weevils?', 'There is no drug available to prevent HIV infection, Pre-exposure prophylaxis (PrEP), yes or no?', 'List two indications of Letermovir?', 'Does allele phasing improve the phylogenetic utility of ultraconserved elements?', 'What is Burning Mouth Syndrome(BMS)?', 'RV3-BB vaccine is used for prevention of which viral infection?', 'What is the role of the Mcm2-Ctf4-Polα axis?', 'What is anophthalmia?', 'Which molecule is inhibited by larotrectinib?', 'What are the roles of LEM-3?', 'What is the most common pediatric glioma?', 'Describe Herpetic Whitlow.', 'List uniparental disomy (UPD) detection algorithms', 'What is Bayesian haplotyping used for?', 'Sweat Chloride Testing is used  for which disease?', 'List the releases of tmVar', 'Does Eucommia ulmoides leaf extract ameliorates steatosis/fatty liver induced by high-fat diet?', 'What is the mechanism of action of Pitolisant?', 'Which methods have been developed for extracting sequence variants from the literature?', 'What is Chrysophanol?', 'Which molecules are targeted by defactinib?', 'What is CardioClassifier?', 'Please list the 4 genes involved in Sanfilippo syndrome, also known as mucopolysaccharidosis III (MPS-III).', 'What is the triad of Melkersson-Rosenthal syndrome?', 'List clinical disorders or diseases where uc.189 is involved?', 'What is the role of metalloproteinase-17 (ADAM17) in NK cells?', 'Is galcanezumab effective for treatment of migraine?', 'Name the algorithms for counting multi-mapping reads', 'Can Diazepam be beneficial  in the treatment of  traumatic brain injury?', 'Is Lasmiditan effective for migraine?', 'Is there a link between BCL11B haploinsufficiency and syndromic neurodevelopmental delay?', 'Fecal transplantation is used to treat infection with what bacteria?', "Is pimavanserin effective for Parkinson's disease psychosis?", 'What are the CADD scores?', 'Erenumab, used to treat migraine headaches, binds to what protein?', 'List drugs included in the TRIUMEQ pill.', 'Which integrin genes are activated by the immune system in inflammatory bowel disease?', 'Which two surgical methods were compared in the RAZOR trial?', 'Which is the database of somatic mutations in normal cells?', 'Which molecule is targeted by upadacitinib?', 'Is Adar3 involved in learning and memory?', 'Which drugs are included in the Lonsurf pill?', 'Mutations in which gene have been found in patients with the CLAPO syndrome?', 'Which enzyme is inhibited by a drug Lorlatinib?', 'Is deletion at 6q24.2-26 associated with longer survival of patients with high-grade serous ovarian carcinoma (HGSOCs)?', 'Describe the mechanism of action of Lurbinectedin.', 'What is CamurWeb?', 'Which enzymes are inhibited by Duvelisib?', 'What is the most common monogenic cause of common variable immunodeficiency (CVID) in Europeans?', 'Is avelumab effective for bladder cancer?', 'De novo mutations in which novel genes are involved in systemic lupus erythematosus?']

'''

'''
[t['query_text'] for t in test_data['queries']]
'''