
from emit_given_text import get_results_for_one_question
from sklearn.preprocessing import MinMaxScaler
from colour import Color
from flask import url_for
import numpy as np
from flask import Flask
from flask import render_template
from flask import request
from collections import OrderedDict

white           = Color("white")
yellow_colors   = list(white.range_to(Color("yellow"), 101))
yellow_colors   = [c.get_hex_l() for c in yellow_colors]
blue_colors     = list(white.range_to(Color("blue"), 101))
blue_colors     = [c.get_hex_l() for c in blue_colors]
green_colors    = list(white.range_to(Color("green"), 101))
green_colors    = [c.get_hex_l() for c in green_colors]

app = Flask(__name__)

@app.route("/")
def get_news():
    return render_template("home.html")

r1 = '''
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.accordion {background-color: #eee; color: #444; cursor: pointer; padding: 18px; width: 100%; border: none; text-align: left; outline: none; font-size: 15px; transition: 0.4s;}
.active, .accordion:hover {background-color: #ccc;}
.panel {padding: 0 18px; display: none; background-color: white; overflow: hidden;}
</style>
</head>
<body>
<title>Results</title>
'''

r2 = '''

<script>
var acc = document.getElementsByClassName("accordion");
var i;

for (i = 0; i < acc.length; i++) {
  acc[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var panel = this.nextElementSibling;
    if (panel.style.display === "block") {
      panel.style.display = "none";
    } else {
      panel.style.display = "block";
    }
  });
}
</script>

</body>
</html>
'''

@app.route("/submit_question", methods=["POST", "GET"])
def submit_question():
    question_text   = request.form.get("sent1") #.strip()
    print(question_text)
    text_to_return  = r1 + '\n' # + r2
    text_to_return  += '<h2>Results for the question: {}</h2>'.format(question_text) + '\n'
    ret_dummy       = get_results_for_one_question(question_text)
    scaler          = MinMaxScaler(feature_range=(0, 0.5))
    scaler.fit(np.array([ret_dummy[doc_id]['doc_score'] for doc_id in ret_dummy]).reshape(-1, 1))
    for doc_id in ret_dummy:
        doc_score       = scaler.transform([[ret_dummy[doc_id]['doc_score']]])[0][0] + 0.5
        doc_bgcolor     = green_colors[int(doc_score*100)]
        doc_txtcolor    = 'white' if(doc_score>0.5) else 'black'
        text_to_return  += '<button title="{}" class="accordion" style="background-color:{};color:{};">PMID:{}</button><div class="panel">'.format(str(doc_score*100), doc_bgcolor, doc_txtcolor, doc_id)
        for sent in ret_dummy[doc_id]['sentences']:
            # print(sent)
            sent_score, sent_text   = sent
            sent_text               = sent_text.replace('</', '< ')
            if(sent_score<0.3):
                sent_score = 0.0
            text_to_return += '<div title="{}" style="width:100%;background-color:{};">{}</div>'.format(sent_score, yellow_colors[int(sent_score*100)], sent_text)
        text_to_return += '<div title="{}" style="width:100%;background-color:{};">{}</div>'.format(
            'link',
            'white',
            'Available on: <a href="https://www.ncbi.nlm.nih.gov/pubmed/?term={}">{}</a>'.format(doc_id, doc_id)
        )
        text_to_return  += '</div>'
    text_to_return += '\n' + r2
    return text_to_return

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

'''
# ret_dummy = OrderedDict([('28705024', {'doc_id': '28705024', 'doc_score': 0.5357710273515364, 'sentences': [(0.6233258247375488, 'MEDI 4736 (durvalumab) in non-small cell lung cancer.'), (0.4008886516094208, 'INTRODUCTION  Immune checkpoint inhibitors (ICI) are now a therapeutic option for advanced non-small cell lung cancer (NSCLC) patients.'), (0.32595688104629517, 'ICI, such as the PD-1 inhibitors nivolumab and pembrolizumab and the PD-L1 inhibitor atezolizumab, have already been marketed for the treatment of pretreated patients with advanced NSCLC.'), (0.5851157307624817, 'Other notable PD-L1 inhibitors under development include avelumab and durvalumab.'), (0.6510307192802429, 'Areas covered: This article reviews literature on durvalumab development, from the preclinical data to the results of phase III clinical trials, whether published or presented at international scientific conferences.'), (0.21979956328868866, 'Ongoing clinical trials were also reviewed.'), (0.6346789598464966, 'Expert opinion: Early phase trials of durvalumab monotherapy (and in combination) have demonstrated activity in advanced NSCLC patients and it has demonstrated a good safety profile.'), (0.7528257369995117, 'The authors believe that durvalumab will likely play an important role in future treatment strategies for NSCLC.'), (0.7110816836357117, 'The PACIFIC trial assessing durvalumab after standard chemoradiotherapy for locally advanced NSCLC has already met its primary endpoint and the potential of durvalumab will be reinforced if phase III randomized studies of first-line (MYSTIC trial) and second or subsequent (ARCTIC trial) lines of therapy demonstrate superiority over the current standard of care.')]}), ('27269937', {'doc_id': '27269937', 'doc_score': 0.19482654811759056, 'sentences': [(0.7294166088104248, 'Safety and Efficacy of Durvalumab (MEDI4736), an Anti-Programmed Cell Death Ligand-1 Immune Checkpoint Inhibitor, in Patients With Advanced Urothelial Bladder Cancer.'), (0.7427939772605896, 'PURPOSE  To investigate the safety and efficacy of durvalumab, a human monoclonal antibody that binds programmed cell death ligand-1 (PD-L1), and the role of PD-L1 expression on clinical response in patients with advanced urothelial bladder cancer (UBC).'), (0.2311149686574936, 'METHODS  A phase 1/2 multicenter, open-label study is being conducted in patients with inoperable or metastatic solid tumors.'), (0.21626651287078857, 'We report here the results from the UBC expansion cohort.'), (0.6826406717300415, 'Durvalumab (MEDI4736, 10 mg/kg every 2 weeks) was administered intravenously for up to 12 months.'), (0.2092062085866928, 'The primary end point was safety, and objective response rate (ORR, confirmed) was a key secondary end point.'), (0.20295563340187073, 'An exploratory analysis of pretreatment tumor biopsies led to defining PD-L1-positive as ≥ 25% of tumor cells or tumor-infiltrating immune cells expressing membrane PD-L1.'), (0.23372237384319305, 'RESULTS  A total of 61 patients (40 PD-L1-positive, 21 PD-L1-negative), 93.4% of whom received one or more prior therapies for advanced disease, were treated (median duration of follow-up, 4.3 months).'), (0.2100621610879898, 'The most common treatment-related adverse events (AEs) of any grade were fatigue (13.1%), diarrhea (9.8%), and decreased appetite (8.2%).'), (0.21049517393112183, 'Grade 3 treatment-related AEs occurred in three patients (4.9%); there were no treatment-related grade 4 or 5 AEs.'), (0.26205453276634216, 'One treatment-related AE (acute kidney injury) resulted in treatment discontinuation.'), (0.20268075168132782, 'The ORR was 31.0% (95% CI, 17.6 to 47.1) in 42 response-evaluable patients, 46.4% (95% CI, 27.5 to 66.1) in the PD-L1-positive subgroup, and 0% (95% CI, 0.0 to 23.2) in the PD-L1-negative subgroup.'), (0.20884910225868225, 'Responses are ongoing in 12 of 13 responding patients, with median duration of response not yet reached (range, 4.1+ to 49.3+ weeks).'), (0.6353549361228943, 'CONCLUSION  Durvalumab demonstrated a manageable safety profile and evidence of meaningful clinical activity in PD-L1-positive patients with UBC, many of whom were heavily pretreated.')]}), ('20156141', {'doc_id': '20156141', 'doc_score': 0.05046075486881049, 'sentences': [(0.41518545150756836, 'Use of compound Chinese medicine in the treatment of lung cancer.'), (0.4295539855957031, 'Traditional Chinese/herbal medicine (TCM) is now commonly used by cancer patients of Asian ethnicity to supplement or replace prescribed treatments.'), (0.40070459246635437, 'The overall survival rate for lung cancer has not improved significantly in the past several decades; it remains the leading cause of cancer death.'), (0.2989906668663025, 'Much more attention has been paid by clinicians and researchers to the possible use of compound Chinese medicine (CCM) as effective anti-lung cancer medicines.'), (0.5147417187690735, 'In this review, we briefly summarize the clinical and experimental status of numerous CCMs recently developed primarily in China for the treatment of lung cancer, including formulations, treatment effectiveness, and molecular mechanisms.'), (0.5533828139305115, 'By presenting this information, our goal is to possibly open up new future avenues for the practice of lung cancer treatment.')]}), ('21514411', {'doc_id': '21514411', 'doc_score': 0.038482143383099056, 'sentences': [(0.4681970477104187, 'Importance of molecular features of non-small cell lung cancer for choice of treatment.'), (0.3932913839817047, 'Lung cancer is the leading cause of cancer-related deaths in the United States.'), (0.37462174892425537, 'Approximately 85% of lung cancer is categorized as non-small cell lung cancer, and traditionally, non-small cell lung cancer has been treated with surgery, radiation, and chemotherapy.'), (0.454976350069046, 'Targeted agents that inhibit the epidermal growth factor receptor pathway have been developed and integrated into the treatment regimens in non-small cell lung cancer.'), (0.22934186458587646, 'Currently, approved epidermal growth factor receptor inhibitors include the tyrosine kinase inhibitors erlotinib and gefitinib.'), (0.5743646621704102, 'Molecular determinants, such as epidermal growth factor receptor-activating mutations, have been associated with response to epidermal growth factor receptor tyrosine kinase inhibitors and may be used to guide treatment choices in patients with non-small cell lung cancer.'), (0.5268421173095703, 'Thus, treatment choice for patients with non-small cell lung cancer depends on molecular features of tumors; however, improved techniques are required to increase the specificity and efficiency of molecular profiling so that these methods can be incorporated into routine clinical practice.'), (0.5870558023452759, 'This review provides an overview of how genetic analysis is currently used to direct treatment choices in non-small cell lung cancer.')]}), ('27822096', {'doc_id': '27822096', 'doc_score': 0.0364512776538096, 'sentences': [(0.340783953666687, 'The Danish Lung Cancer Registry.'), (0.32194003462791443, 'AIM OF DATABASE  The Danish Lung Cancer Registry (DLCR) was established by the Danish Lung Cancer Group.'), (0.3419380187988281, 'The primary and first goal of the DLCR was to improve survival and the overall clinical management of Danish lung cancer patients.'), (0.34481531381607056, 'STUDY POPULATION  All Danish primary lung cancer patients since 2000 are included into the registry and the database today contains information on more than 50,000 cases of lung cancer.'), (0.3549940288066864, 'MAIN VARIABLES  The database contains information on patient characteristics such as age, sex, diagnostic procedures, histology, tumor stage, lung function, performance, comorbidities, type of surgery, and/or oncological treatment and complications.'), (0.2422512322664261, 'Since November 2013, DLCR data on Patient -Reported Outcome Measures is also included.'), (0.23813496530056, 'DESCRIPTIVE DATA  Results are primarily reported as quality indicators, which are published online monthly, and in an annual report where the results are commented for local, regional, and national audits.'), (0.2725444734096527, 'Indicator results are supported by descriptive reports with details on diagnostics and treatment.'), (0.643467903137207, 'CONCLUSION  DLCR has since its creation been used to improve the quality of treatment of lung cancer in Denmark and it is increasingly used as a source for research regarding lung cancer in Denmark and in comparisons with other countries.')]}), ('23462173', {'doc_id': '23462173', 'doc_score': 0.029696672393742164, 'sentences': [(0.35216131806373596, 'Bovine lactoferrin inhibits lung cancer growth through suppression of both inflammation and expression of vascular endothelial growth factor.'), (0.4071521461009979, 'Lung cancers are among the most common cancers in the world, and the search for effective and safe drugs for the chemoprevention and therapy of pulmonary cancer has become important.'), (0.43646055459976196, 'In this study, bovine lactoferrin (bLF) was used in both in vitro and in vivo approaches to investigate its activity against lung cancer.'), (0.5897518396377563, 'A human lung cancer cell line, A549, which expresses a high level of vascular endothelial growth factor (VEGF) under hypoxia, was used as an in vitro system for bLF treatment.'), (0.4458851218223572, 'A strain of transgenic mice carrying the human VEGF-A165 (hVEGF-A165) gene, which induces pulmonary tumors, was used as an in vivo lung cancer therapy model.'), (0.20753443241119385, 'We found that bLF significantly decreased proliferation of A549 cells by decreasing the expression of VEGF protein in a dose-dependent manner.'), (0.24200405180454254, 'Furthermore, oral administration of bLF at 300 mg/kg of body weight 3 times a week for 1.5 mo to the transgenic mice overexpressing hVEGF-A165 significantly eliminated expression of hVEGF-A165 and suppressed the formation of tumors.'), (0.2758330702781677, 'Additionally, treatment with bLF significantly decreased the levels of proinflammatory cytokines, such as tumor necrosis factor-α, and antiinflammatory cytokines, such as IL-4 and IL-10.'), (0.23588570952415466, 'Levels of IL-6, which is both a proinflammatory and an antiinflammatory cytokine, were also reduced.'), (0.4477081894874573, 'Treatment with bLF decreased levels of tumor necrosis factor-α, IL-4, IL-6, and IL-10 cytokines, resulting in limited inflammation, which then restricted growth of the lung cancer.'), (0.5437588691711426, 'Our results revealed that bLF is an inhibitor of angiogenesis and blocks lung cell inflammation; as such, it has considerable potential for therapeutic use in the treatment of lung cancer.')]}), ('9209786', {'doc_id': '9209786', 'doc_score': 0.029695652856908967, 'sentences': [(0.42503121495246887, 'Significant survival benefit to patients with advanced non-small-cell lung cancer from treatment with clarithromycin.'), (0.36839985847473145, 'We carried out a randomized study of 49 consecutive patients with unresectable primary lung cancer to determine whether clarithromycin (CAM), a 14-membered ring macrolide, can improve outcome.'), (0.33949950337409973, 'A total of 49 patients (42 patients with non-small-cell lung cancer and 7 patients with small-cell lung cancer) had received prior chemotherapy, radiotherapy or both during their hospital stay.'), (0.35980024933815, 'They were randomly allocated into two study groups on the first visit after discharge: 25 patients (22 patients with non-small-cell lung cancer, 3 patients with small-cell lung cancer) were assigned to receive CAM (400 mg/day, orally), and 24 patients (20 patients with non-small-cell lung cancer, 4 patients with small-cell lung cancer) did not receive CAM.'), (0.27459293603897095, 'CAM treatment after randomization was open and the treatment was to be continued as long as the patients could tolerate CAM.'), (0.394532710313797, 'There was no significant difference in the median survival time for small-cell lung cancer between the CAM group and the non-CAM group.'), (0.50352543592453, 'However, CAM treatment significantly increased the median survival time for non-small-cell lung cancer patients, the median survival for the CAM group was 535 days and that for the non-CAM group was 277 days.'), (0.5006833672523499, 'Analyses of prognostic factors showed that only treatment with CAM was predictive of longer survival for non-small-cell lung cancer, and other tested covariates had no effects on the prognosis.'), (0.26566198468208313, 'There were no remarkable side effects observed in the CAM group throughout treatment.'), (0.5568233132362366, 'We conclude that long-term treatment using CAM is beneficial for unresectable non-small-cell lung cancer patients and that it can increase the median survival of patients with advanced disease.')]}), ('18979097', {'doc_id': '18979097', 'doc_score': 0.029351388466351384, 'sentences': [(0.4608594477176666, 'HM1.24 (CD317) is a novel target against lung cancer for immunotherapy using anti-HM1.24 antibody.'), (0.24573545157909393, 'HM1.24 antigen (CD317) was originally identified as a cell surface protein that is preferentially overexpressed on multiple myeloma cells.'), (0.21410517394542694, 'Immunotherapy using anti-HM1.24 antibody has been performed in patients with multiple myeloma as a phase I study.'), (0.37037673592567444, 'We examined the expression of HM1.24 antigen in lung cancer cells and the possibility of immunotherapy with anti-HM1.24 antibody which can induce antibody-dependent cellular cytotoxicity (ADCC).'), (0.35723090171813965, 'The expression of HM1.24 antigen in lung cancer cells was examined by flow cytometry as well as immunohistochemistry using anti-HM1.24 antibody.'), (0.2246146947145462, 'ADCC was evaluated using a 6-h (51)Cr release assay.'), (0.21269898116588593, 'Effects of various cytokines on the expression of HM1.24 and the ADCC were examined.'), (0.21318595111370087, 'The antitumor activity of anti-HM1.24 antibody in vivo was examined in SCID mice.'), (0.32843467593193054, 'HM1.24 antigen was detected in 11 of 26 non-small cell lung cancer cell lines (42%) and four of seven (57%) of small cell lung cancer cells, and also expressed in the tissues of lung cancer.'), (0.35292160511016846, 'Anti-HM1.24 antibody effectively induced ADCC in HM1.24-positive lung cancer cells.'), (0.3528064489364624, 'Interferon-beta and -gamma increased the levels of HM1.24 antigen and the susceptibility of lung cancer cells to ADCC.'), (0.44603288173675537, 'Treatment with anti-HM1.24 antibody inhibited the growth of lung cancer cells expressing HM1.24 antigen in SCID mice.'), (0.27530354261398315, 'The combined therapy with IFN-beta and anti-HM1.24 antibody showed the enhanced antitumor effects even in the delayed treatment schedule.'), (0.5490902066230774, 'HM1.24 antigen is a novel immunological target for the treatment of lung cancer with anti-HM1.24 antibody.')]}), ('8113100', {'doc_id': '8113100', 'doc_score': 0.02887542864412835, 'sentences': [(0.26398083567619324, 'Dose-volume histogram and 3-D treatment planning evaluation of patients with pneumonitis.'), (0.29768767952919006, 'PURPOSE  Tolerance of normal lung to inhomogeneous irradiation of partial volumes is not well understood.'), (0.39008641242980957, 'This retrospective study analyzes three-dimensional (3-D) dose distributions and dose-volume histograms for 63 patients who have had normal lung irradiated in two types of treatment situations.'), (0.48849624395370483, "METHODS AND MATERIALS  3-D treatment plans were examined for 21 patients with Hodgkin's disease and 42 patients with nonsmall-cell lung cancer."), (0.3932313621044159, 'All patients were treated with conventional fractionation, with a dose of 67 Gy (corrected) or higher for the lung cancer patients.'), (0.2676668167114258, 'A normal tissue complication probability description and a dose-volume histogram reduction scheme were used to assess the data.'), (0.2692404091358185, 'Mean dose to lung was also calculated.'), (0.33479616045951843, "RESULTS  Five Hodgkin's disease patients and nine lung cancer patients developed pneumonitis."), (0.2522142231464386, 'Data were analyzed for each individual independent lung and for the total lung tissue (lung as a paired organ).'), (0.2528674304485321, 'Comparisons of averages of mean lung dose and normal tissue complication probabilities show a difference between patients with and without complications.'), (0.4234110713005066, "Averages of calculated normal tissue complication probabilities for groups of patients show that empirical model parameters correlate with actual complication rates for the Hodgkin's patients, but not as well for the individual lungs of the lung cancer patients treated to larger volumes of normal lung and high doses."), (0.39229169487953186, 'CONCLUSION  This retrospective study of the 3-D dose distributions for normal lung for two types of treatment situations for patients with irradiated normal lung gives useful data for the characterization of the dose-volume relationship and the development of pneumonitis.'), (0.5823609828948975, 'These data can be used to help set up a dose escalation protocol for the treatment of nonsmall-cell lung cancer.')]}), ('26346948', {'doc_id': '26346948', 'doc_score': 0.026389106264022945, 'sentences': [(0.4503554403781891, 'Readiness of Lung Cancer Screening Sites to Deliver Smoking Cessation Treatment: Current Practices, Organizational Priority, and Perceived Barriers.'), (0.35716530680656433, 'INTRODUCTION  Lung cancer screening represents an opportunity to deliver smoking cessation advice and assistance to current smokers.'), (0.4341793358325958, 'However, the current tobacco treatment practices of lung cancer screening sites are unknown.'), (0.5165449380874634, 'The purpose of this study was to describe organizational priority, current practice patterns, and barriers for delivery of evidence-based tobacco use treatment across lung cancer screening sites within the United States.'), (0.47176435589790344, 'METHODS  Guided by prior work examining readiness of health care providers to deliver tobacco use treatment, we administered a brief online survey to a purposive national sample of site coordinators from 93 lung cancer screening sites.'), (0.3980845808982849, 'RESULTS  Organizational priority for promoting smoking cessation among lung cancer screening enrollees was high.'), (0.2121080905199051, 'Most sites reported that, at the initial visit, patients are routinely asked about their current smoking status (98.9%) and current smokers are advised to quit (91.4%).'), (0.2148025780916214, 'Fewer (57%) sites provide cessation counseling or refer smokers to a quitline (60.2%) and even fewer (36.6%) routinely recommend cessation medications.'), (0.27938172221183777, 'During follow-up screening visits, respondents reported less attention to smoking cessation advice and treatment.'), (0.33188703656196594, 'Lack of patient motivation and resistance to cessation advice and treatment, lack of staff training, and lack of reimbursement were the most frequently cited barriers for delivering smoking cessation treatment.'), (0.5788857936859131, 'CONCLUSIONS  Although encouraging that lung cancer screening sites endorsed the importance of smoking cessation interventions, greater attention to identifying and addressing barriers for tobacco treatment delivery is needed in order to maximize the potential benefit of integrating smoking cessation into lung cancer screening protocols.'), (0.5648910403251648, 'IMPLICATIONS  This study is the first to describe practice patterns, organizational priority, and barriers for delivery of smoking cessation treatment in a national sample of lung cancer screening sites.')]})])
'''
