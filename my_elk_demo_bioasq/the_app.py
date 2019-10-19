
# from .emit_given_text import get_results_for_one_question
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
<h2>Results</h2>
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

ret_dummy = OrderedDict(
    [
        ('28705024',
              {'doc_id': '28705024',
               'doc_score': 0.126079710986306,
               'sentences': [(0.7528257369995117,
                              'The authors believe that durvalumab will likely '
                              'play an important role in future treatment '
                              'strategies for NSCLC.'),
                             (0.7110816836357117,
                              'The PACIFIC trial assessing durvalumab after '
                              'standard chemoradiotherapy for locally advanced '
                              'NSCLC has already met its primary endpoint and '
                              'the potential of durvalumab will be reinforced '
                              'if phase III randomized studies of first-line '
                              '(MYSTIC trial) and second or subsequent (ARCTIC '
                              'trial) lines of therapy demonstrate superiority '
                              'over the current standard of care.'),
                             (0.6510307192802429,
                              'Areas covered: This article reviews literature '
                              'on durvalumab development, from the preclinical '
                              'data to the results of phase III clinical '
                              'trials, whether published or presented at '
                              'international scientific conferences.'),
                             (0.6346789598464966,
                              'Expert opinion: Early phase trials of '
                              'durvalumab monotherapy (and in combination) '
                              'have demonstrated activity in advanced NSCLC '
                              'patients and it has demonstrated a good safety '
                              'profile.'),
                             (0.6233258247375488,
                              'MEDI 4736 (durvalumab) in non-small cell lung '
                              'cancer.'),
                             (0.5851157307624817,
                              'Other notable PD-L1 inhibitors under '
                              'development include avelumab and durvalumab.'),
                             (0.4008886516094208,
                              'INTRODUCTION  Immune checkpoint inhibitors '
                              '(ICI) are now a therapeutic option for advanced '
                              'non-small cell lung cancer (NSCLC) patients.'),
                             (0.32595688104629517,
                              'ICI, such as the PD-1 inhibitors nivolumab and '
                              'pembrolizumab and the PD-L1 inhibitor '
                              'atezolizumab, have already been marketed for '
                              'the treatment of pretreated patients with '
                              'advanced NSCLC.'),
                             (0.21979956328868866,
                              'Ongoing clinical trials were also reviewed.')]}),
        ('28512504',
          {'doc_id': '28512504',
           'doc_score': 0.11415850781104293,
           'sentences': [(0.8161256313323975,
                          'A Case Report of Drug-Induced Myopathy '
                          'Involving Extraocular Muscles after Combination '
                          'Therapy with Tremelimumab and Durvalumab for '
                          'Non-Small Cell Lung Cancer.'),
                         (0.8108379244804382,
                          'The authors report a 68-year-old man treated '
                          'for non-small cell lung cancer (NSCLC) with a '
                          'combination of tremelimumab and durvalumab.'),
                         (0.6252461671829224,
                          'Recently developed anti-tumour therapies '
                          'targeting immune checkpoints include '
                          'tremelimumab and durvalumab.'),
                         (0.28304508328437805,
                          'Within 1 month of withdrawal of cancer '
                          'therapies and initiation of oral steroid '
                          'therapy, ocular and systemic symptoms had '
                          'resolved.'),
                         (0.2772778570652008,
                          'After treatment he developed diplopia, ptosis, '
                          'fatigue, weakness, and an inflammatory myopathy '
                          'affecting the extraocular muscles requiring '
                          'hospitalisation.'),
                         (0.24444884061813354,
                          'This notable adverse effect has not been '
                          'previously described for these drugs '
                          'administered singly or in combination, and '
                          'ophthalmologists should be aware of this '
                          'presentation in patients treated with these '
                          'agents.'),
                         (0.24131029844284058,
                          'These agents have incompletely characterised '
                          'side effect profiles.'),
                         (0.21184565126895905,
                          'Electromyography (EMG) testing and muscle '
                          'biopsy suggested inflammatory myopathy without '
                          'sign of myasthenia.')]}),
        ('28963357',
        {'doc_id': '28963357',
           'doc_score': 0.11366683069742717,
           'sentences': [(0.7939553260803223,
                          'The PD-L1 inhibitor durvalumab increases '
                          'progression-free survival and objective '
                          'response rate in patients with inoperable and '
                          'locally advanced stage III non-small cell lung '
                          'cancer, according to interim results of a phase '
                          'III trial.'),
                         (0.5450866222381592,
                          'Durvalumab Promising for NSCLC.'),
                         (0.22926607728004456,
                          'The benefit was great enough that the drug '
                          'could become the standard of care in the United '
                          'States for these patients.')]}),
        ('28585617',
          {'doc_id': '28585617',
           'doc_score': 0.10832927074395596,
           'sentences': [(0.5606199502944946,
                          'Similarly to other malignancies, immune '
                          'checkpoint inhibitor therapy is a '
                          'revolutionary, effective new treatment '
                          'possibility for lung cancer.'),
                         (0.5580491423606873,
                          'Avelumab and durvalumab have promising activity '
                          'as well.'),
                         (0.41366899013519287,
                          'The PD-1 inhibitor nivolumab and pembrolizumab '
                          'and the PD-L1 inhibitor atezolizumab is a '
                          'labelled indication in second line setting in '
                          'advanced nonsmall cell lung cancer (NSCLC).'),
                         (0.39814069867134094,
                          'In lung cancer carcinogenesis is related mainly '
                          'to tobacco smoking with high somatic mutation '
                          'rate and immunogenicity.'),
                         (0.3645658493041992,
                          'There are numerous ongoing clinical trials in '
                          'lung cancer with immune checkpoint inhibitors '
                          'in combination with cytotoxic chemotherapy, '
                          'targeted agents, or in adjuvant setting.'),
                         (0.36173710227012634,
                          '[Immunotherapy for lung cancer].'),
                         (0.28228625655174255,
                          'Based on the data of KEYNOTE 024 trial, '
                          'pembrolizumab is approved in first line setting '
                          'for cases with ≥50% PD-L1 expression.'),
                         (0.26311177015304565,
                          'Pembrolizumab treatment became a new first line '
                          'standard of care in advanced NSCLC.'),
                         (0.20108452439308167,
                          'In this selected patient population, '
                          'progression-free survival has doubled, and '
                          'overall survival was significantly better in '
                          'pembrolizumab-treated patients compared to '
                          'those receiving standard of care.')]}),
        ('27019997',
          {'doc_id': '27019997',
           'doc_score': 0.09715932456508534,
           'sentences': [(0.6930536031723022,
                          'The efficacy and safety data for drugs such as '
                          'ipilimumab, nivolumab, pembrolizumab, '
                          'atezolizumab and durvalumab are reviewed, along '
                          'with combination strategies and response '
                          'evaluation criteria.'),
                         (0.5222896933555603,
                          'Immune checkpoint inhibitors targeting CTLA-4, '
                          'PD-1 and PD-L1 have been developed for the '
                          'treatment of patients with non-small-cell lung '
                          'cancer and other malignancies, with impressive '
                          'clinical activity, durable responses and a '
                          'favorable toxicity profile.'),
                         (0.34059178829193115,
                          'Immune checkpoint inhibitors in lung cancer: '
                          'past, present and future.'),
                         (0.3147279620170593,
                          'Inhibitory ligands on tumor cells and their '
                          'corresponding receptors on T cells are '
                          'collectively called immune checkpoint molecules '
                          'and have emerged as druggable targets that '
                          'harness endogenous immunity to fight cancer.'),
                         (0.23520700633525848,
                          'This article reviews the development, current '
                          'status and future directions for some of these '
                          'agents.'),
                         (0.2127751260995865,
                          'The toxicity profiles and predictive biomarkers '
                          'of response are also discussed.')]}),
        ('26882955',
              {'doc_id': '26882955',
               'doc_score': 0.09481262404830765,
               'sentences': [(0.797855019569397,
                              'Immune checkpoint inhibitors such as '
                              'ipilimumab, nivolumab, pembrolizumab, '
                              'durvalumab, tremelimumab and ulocuplumab are at '
                              'the forefront of immunotherapy and have '
                              'achieved approvals for certain cancer types, '
                              'including melanoma (ipilimumab, nivolumab and '
                              'pembrolizumab), non-SCLC (nivolumab and '
                              'pembrolizumab) and renal cell carcinoma '
                              '(nivolumab).'),
                             (0.498870849609375,
                              'Treatment for small-cell lung cancer (SCLC) has '
                              'changed little over the past few decades; '
                              'available therapies have failed to extend '
                              'survival in advanced disease.'),
                             (0.3787004053592682,
                              'Immunotherapy for small-cell lung cancer: '
                              'emerging evidence.'),
                             (0.2802731394767761,
                              'In recent years, immunotherapy with treatments '
                              'such as interferons, TNFs, vaccines and immune '
                              'checkpoint inhibitors has advanced and shown '
                              'promise in the treatment of several tumor '
                              'types.'),
                             (0.21201610565185547,
                              'We review emerging evidence supporting the use '
                              'of immunotherapy in SCLC patients.'),
                             (0.20732387900352478,
                              'Clinical trials are investigating different '
                              'immunotherapies in patients with other solid '
                              'and hematologic malignancies, including '
                              'SCLC.')]}),
        ('27532023',
              {'doc_id': '27532023',
               'doc_score': 0.09070824008449527,
               'sentences': [(0.7084934711456299,
                              'Two drugs, nivolumab and pembrolizumab, are now '
                              'FDA approved for use in certain patients who '
                              'have failed or progressed on platinum-based or '
                              'targeted therapies while agents targeting '
                              'PD-L1, atezolizumab and durvalumab, are '
                              'approaching the final stages of clinical '
                              'testing.'),
                             (0.4696488678455353,
                              'Various monoclonal antibodies which block the '
                              'interaction between checkpoint molecules PD-1 '
                              'on immune cells and PD-L1 on cancer cells have '
                              'been used to successfully treat non-small cell '
                              'lung cancer (NSCLC), including some durable '
                              'responses lasting years.'),
                             (0.38527911901474,
                              'PD-L1 biomarker testing for non-small cell lung '
                              'cancer: truth or fiction?'),
                             (0.33129963278770447,
                              'Research in cancer immunology is currently '
                              'accelerating following a series of cancer '
                              'immunotherapy breakthroughs during the last '
                              '5\xa0years.'),
                             (0.28581470251083374,
                              'Despite impressive treatment outcomes in a '
                              'subset of patients who receive these immune '
                              'therapies, many patients with NSCLC fail to '
                              'respond to anti-PD-1/PD-L1 and the '
                              'identification of a biomarker to select these '
                              'patients remains highly sought after.'),
                             (0.23212508857250214,
                              'In this review, we discuss the recent clinical '
                              'trial results of pembrolizumab, nivolumab, and '
                              'atezolizumab for NSCLC, and the significance of '
                              'companion diagnostic testing for tumor PD-L1 '
                              'expression.')]}),
        ('27389724',
              {'doc_id': '27389724',
               'doc_score': 0.08582881266429562,
               'sentences': [(0.6827545166015625,
                              'Despite not being eligible for a durvalumab '
                              'trial because of lack of PD-L1 expression, she '
                              'had a meaningful response to nivolumab.'),
                             (0.3382163345813751,
                              'A 58-year-old woman, a heavy smoker, was '
                              'diagnosed with stage III squamous cell lung '
                              'cancer.'),
                             (0.33626845479011536,
                              'Battling regional (stage III) lung cancer: '
                              'bumpy road of a cancer survivor in the '
                              'immunotherapy age.'),
                             (0.2854982912540436,
                              'She had disease progression 9\u2005months into '
                              'treatment.'),
                             (0.24585649371147156,
                              '4\u2005months later, a rapidly enlarging renal '
                              'mass was discovered and turned out to be '
                              'metastatic from the lung primary.'),
                             (0.24071486294269562,
                              'Fortunately, her bleed was self-limited.'),
                             (0.2161630392074585,
                              '2\u2005months later, she had haemoptysis caused '
                              'by brisk bleeding from the radiated right upper '
                              'lobe.'),
                             (0.212578684091568,
                              'She was treated with concurrent chemotherapy '
                              'and radiotherapy, with partial response.'),
                             (0.21097226440906525,
                              'In the next few months, she experienced a '
                              'variety of side effects, some of which were '
                              'potentially life-threatening.'),
                             (0.21042422950267792,
                              'Second-line chemotherapy with docetaxel and '
                              'ramucirumab did not have effects on the renal '
                              'mass after 2 cycles.'),
                             (0.20962224900722504,
                              'Once every 2\u2005weeks, infusion of nivolumab '
                              'resulted in rapid tumour shrinkage in multiple '
                              'areas.')]}),
        ('28271729',
              {'doc_id': '28271729',
               'doc_score': 0.08509885116571422,
               'sentences': [(0.7143888473510742,
                              'The combination of osimertinib plus durvalumab '
                              'in pretreated or chemo naïve NSCLC patients '
                              'showed encouraging clinical activity, however, '
                              'this combination was associated with high '
                              'incidence of interstitial lung disease (38%), '
                              'leading to termination of further enrollment.'),
                             (0.6330975890159607,
                              'The combination of gefitinib plus durvalumab '
                              'demonstrated encouraging activity but higher '
                              'incidence of grade 3/4 liver enzyme elevation '
                              '(40-70%).'),
                             (0.3779904842376709,
                              'INTRODUCTION  Epidermal growth factor receptor '
                              '(EGFR) tyrosine kinase inhibitors (TKI) has '
                              'significantly improved clinical outcomes '
                              'compared with chemotherapy in non-small cell '
                              'lung cancer (NSCLC) patients with sensitizing '
                              'EGFR gene mutation.'),
                             (0.3333910405635834,
                              'EGFR TKI combination with immunotherapy in '
                              'non-small cell lung cancer.'),
                             (0.2777625322341919,
                              'It has been reported that activation of the '
                              'oncogenic EGFR pathway enhances susceptibility '
                              'of the lung tumors to PD-1 blockade in mouse '
                              'model, suggesting combination of PD1 blockade '
                              'with EGFR TKIs may be a promising therapeutic '
                              'strategy.'),
                             (0.26722806692123413,
                              'The treatment related Grade 3-4 adverse events '
                              'were observed in 39% of patients when treated '
                              'with atezolizumab plus erlotinib.'),
                             (0.2256692498922348,
                              'Until now, the combination of EGFR TKI and '
                              'immunotherapy should be investigational.'),
                             (0.21707475185394287,
                              'Areas covered: Almost all patients treated with '
                              'EGFR TKIs eventually develop acquired '
                              'resistance.'),
                             (0.21395689249038696,
                              'Nivolumab combined with erlotinib was '
                              'associated with 19% of grade 3 toxicities.'),
                             (0.20948049426078796,
                              'Expert opinion: Given the relatively high '
                              'incidence of treatment-related toxicities '
                              'associated with combination of EGFR TKI and '
                              'immunotherapy, further development of this '
                              'approach remains controversial.')]}),
        ('28064556',
              {'doc_id': '28064556',
               'doc_score': 0.08415782723336986,
               'sentences': [(0.69700688123703,
                              'The authors discuss pembrolizumab and '
                              'pembrolizumab plus ipilimumab, durvalumab and '
                              'durvalumab plus tremelimumab, nivolumab and '
                              'nivolumab plus ipilimumab for NSCLC as well as '
                              'nivolumab and nivolumab plus ipilimumab for '
                              'SCLC.'),
                             (0.41634857654571533,
                              'The increased response rate, if confirmed in '
                              'larger scale studies, will likely make '
                              'combination therapy another useful therapeutic '
                              'approach for lung cancer.'),
                             (0.4094315767288208,
                              'PD-1 checkpoint blockade alone or combined PD-1 '
                              'and CTLA-4 blockade as immunotherapy for lung '
                              'cancer?'),
                             (0.36628714203834534,
                              'Blocking a single immune checkpoint or multiple '
                              'checkpoints simultaneously can generate '
                              'anti-tumor activity against a variety of '
                              'cancers including lung cancer.'),
                             (0.34728947281837463,
                              'Area covered: This review highlights the '
                              'results of recent clinical studies of single or '
                              'combination checkpoint inhibitor immunotherapy '
                              'in non-small cell lung cancer (NSCLC) or small '
                              'cell lung cancer (SCLC).'),
                             (0.2873440086841583,
                              'For checkpoint inhibitor immunotherapy in SCLC '
                              'and NSCLC, combination therapy is associated '
                              'with a higher incidence of toxicities than '
                              'single therapy; however, it appears to help '
                              'increase tumor response rate.'),
                             (0.2525465190410614,
                              'Nevertheless, combination therapy is associated '
                              'with an increased toxicity.'),
                             (0.23285989463329315,
                              'Several larger-scale studies are currently '
                              'ongoing.'),
                             (0.22378256916999817,
                              'INTRODUCTION  Signaling through T-cell surface, '
                              'an immune checkpoint protein such as PD-1 or '
                              'CTLA-4 helps dampen or terminate unwanted '
                              'immune responses.'),
                             (0.2055046707391739,
                              'Expert opinion: Available data suggest that, in '
                              'both metastatic NSCLC and SCLC, combined PD-1 '
                              'and CTLA-4 blockade may produce a higher tumor '
                              'response rate than PD-1 blockade alone.')]})
    ]
)

@app.route("/submit_question", methods=["POST", "GET"])
def submit_question():
    text_to_return = r1 + '\n' # + r2
    # print(request.form)
    question_text   = request.form.get("sent1") #.strip()
    scaler          = MinMaxScaler(feature_range=(0, 0.5))
    scaler.fit(np.array([ret_dummy[doc_id]['doc_score'] for doc_id in ret_dummy]).reshape(-1, 1))
    for doc_id in ret_dummy:
        doc_score       = scaler.transform([[ret_dummy[doc_id]['doc_score']]])[0][0] + 0.5
        doc_bgcolor     = green_colors[int(doc_score*100)]
        doc_txtcolor    = 'white' if(doc_score>0.5) else 'black'
        text_to_return  += '<button title="{}" class="accordion" style="background-color:{};color:{};">PMID:{}</button><div class="panel">'.format(str(doc_score*100), doc_bgcolor, doc_txtcolor, doc_id)
        for sent in ret_dummy[doc_id]['sentences']:
            print(sent)
            sent_score, sent_text = sent
            text_to_return += '<div style="width:100%;background-color:{};">{}</div>'.format(yellow_colors[int(sent_score*100)], sent_text)
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