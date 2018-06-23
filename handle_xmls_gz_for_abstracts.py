#!/usr/bin/env python

'''

# download files from
# https://mbr.nlm.nih.gov/Download/Baselines/

for i in range(1,893):
    print 'wget https://mbr.nlm.nih.gov/Download/Baselines/2017/medline17n{0:0>4}.xml.gz'.format(i)

'''


import os
import gzip
import traceback
from lxml import etree
from pprint import pprint
from dateutil import parser
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch import helpers
import random


es = Elasticsearch(['localhost:9200'], verify_certs=True, timeout=150, max_retries=10, retry_on_timeout=True)

index = 'pubmed_abstracts_index_0_1'
doc_type = "pubmed_abstracts_mapping_0_1"

def abs_found(pmid):
    bod = {"query" : {'bool':{"must" : [ { "term": { "pmid": pmid } }]}}}
    res = es.search(index=index, doc_type=doc_type, body=bod)
    return len(res['hits']['hits'])>0

def create_new_xml_from_element(element):
    return etree.fromstring(etree.tostring(element))

def get_children_with_tag(elem,tag):
    return [ x for x in elem.getchildren() if(x.tag == tag) and x is not None ]

def get_OtherIDs():
    OtherIDs = get_children_with_tag(elem,'OtherID')
    dato['OtherIDs'] = [
        {
            'Source'    :id.get('Source').strip(),
            'id'        :id.text.strip(),
        } for id in OtherIDs
    ]
    for OtherID in OtherIDs: OtherID.getparent().remove(OtherID)

def get_MeshHeadings():
    MeshHeadingList = get_children_with_tag(elem, 'MeshHeadingList')
    if(len(MeshHeadingList)>0):
        MeshHeadingList = MeshHeadingList[0]
        MeshHeadings = get_children_with_tag(MeshHeadingList, 'MeshHeading')
        dato['MeshHeadings'] = []
        for MeshHeading in MeshHeadings:
            mh = {}
            DescriptorName = get_children_with_tag(MeshHeading, 'DescriptorName')
            if(len(DescriptorName)>0):
                DescriptorName = DescriptorName[0]
                mh['DescriptorName'] = {
                    'text'          : DescriptorName.text.strip(),
                    'UI'            : DescriptorName.get('UI').strip(),
                    # 'MajorTopicYN'  : DescriptorName.get('MajorTopicYN').strip(),
                    'Type'          : DescriptorName.get('Type').strip() if (DescriptorName.get('Type') is not None) else '',
                }
            QualifierName = get_children_with_tag(MeshHeading, 'QualifierName')
            if(len(QualifierName)>0):
                QualifierName = QualifierName[0]
                mh['QualifierName'] = {
                    'text'          : QualifierName.text.strip(),
                    'UI'            : QualifierName.get('UI').strip(),
                    # 'MajorTopicYN'  : QualifierName.get('MajorTopicYN').strip(),
                    'Type'          : QualifierName.get('Type').strip() if (QualifierName.get('Type') is not None) else '',
                }
            dato['MeshHeadings'].append(mh)
            MeshHeading.getparent().remove(MeshHeading)
        MeshHeadingList.getparent().remove(MeshHeadingList)

def get_Authors():
    AuthorList = get_children_with_tag(Article, 'AuthorList')
    if(len(AuthorList)>0):
        AuthorList = AuthorList[0]
        Authors     = get_children_with_tag(AuthorList, 'Author')
        dato['Authors'] = []
        for Author in Authors:
            au = {
                    'LastName': get_children_with_tag(Author, 'LastName')[0].text.strip() if (len(get_children_with_tag(Author, 'LastName'))>0) else '',
                    'ForeName': get_children_with_tag(Author, 'ForeName')[0].text.strip() if (len(get_children_with_tag(Author, 'ForeName'))>0) else '',
                    'Initials': get_children_with_tag(Author, 'Initials')[0].text.strip() if (len(get_children_with_tag(Author, 'Initials'))>0) else '',
                    'AffiliationInfo': get_children_with_tag(Author, 'AffiliationInfo')[0].text.strip() if (len(get_children_with_tag(Author, 'AffiliationInfo'))>0) else '',
                    'CollectiveName': get_children_with_tag(Author, 'CollectiveName')[0].text.strip() if (len(get_children_with_tag(Author, 'CollectiveName'))>0) else '',
            }
            dato['Authors'].append(au)
            Author.getparent().remove(Author)
        AuthorList.getparent().remove(AuthorList)

def get_PersonalNameSubjectList():
    PersonalNameSubjectList = get_children_with_tag(elem, 'PersonalNameSubjectList')
    if(len(PersonalNameSubjectList)>0):
        PersonalNameSubjectList = PersonalNameSubjectList[0]
        PersonalNameSubjects     = get_children_with_tag(PersonalNameSubjectList, 'PersonalNameSubjects')
        dato['PersonalNameSubjects'] = []
        for PersonalNameSubject in PersonalNameSubjects:
            au = {
                    'LastName': get_children_with_tag(PersonalNameSubject, 'LastName')[0].text.strip() if (len(get_children_with_tag(PersonalNameSubject, 'LastName'))>0) else '',
                    'ForeName': get_children_with_tag(PersonalNameSubject, 'ForeName')[0].text.strip() if (len(get_children_with_tag(PersonalNameSubject, 'ForeName'))>0) else '',
                    'Initials': get_children_with_tag(PersonalNameSubject, 'Initials')[0].text.strip() if (len(get_children_with_tag(PersonalNameSubject, 'Initials'))>0) else '',
            }
            dato['PersonalNameSubjects'].append(au)
            PersonalNameSubject.getparent().remove(PersonalNameSubject)
        PersonalNameSubjectList.getparent().remove(PersonalNameSubjectList)

def get_Abstract():
    Abstract = get_children_with_tag(Article, 'Abstract')
    dato['AbstractText'] = []
    if(len(Abstract)>0):
        Abstract = Abstract[0]
        for AbstractText in get_children_with_tag(Abstract, 'AbstractText'):
            dato['AbstractText'].append(
                {

                    'text'          : get_element_lower_text(AbstractText).strip(),
                    'Label'         : AbstractText.get('Label').strip() if ('Label' in AbstractText.keys()) else '',
                    'NlmCategory'   : AbstractText.get('NlmCategory').strip() if ('NlmCategory' in AbstractText.keys()) else '',
                }
            )
            AbstractText.getparent().remove(AbstractText)
        Abstract.getparent().remove(Abstract)

def get_ArticleDate():
    ArticleDate = get_children_with_tag(Article, 'ArticleDate')
    if(len(ArticleDate)>0):
        ArticleDate = ArticleDate[0]
        dato['ArticleDate']  = get_children_with_tag(ArticleDate,'Day')[0].text.strip() + '/' \
                               +get_children_with_tag(ArticleDate,'Month')[0].text.strip() + '/' \
                               +get_children_with_tag(ArticleDate,'Year')[0].text.strip()
        ArticleDate.getparent().remove(ArticleDate)

def get_MedlineJournalInfo():
    MedlineJournalInfo = get_children_with_tag(elem, 'MedlineJournalInfo')[0]
    dato['MedlineJournalInfo'] = {
        'Country'       : get_children_with_tag(MedlineJournalInfo, 'Country')[0].text.strip(),
        'MedlineTA'     : get_children_with_tag(MedlineJournalInfo, 'MedlineTA')[0].text.strip(),
        'NlmUniqueID'   : get_children_with_tag(MedlineJournalInfo, 'NlmUniqueID')[0].text.strip(),
        'ISSNLinking'   : get_children_with_tag(MedlineJournalInfo, 'ISSNLinking')[0].text.strip() if (len(get_children_with_tag(MedlineJournalInfo, 'ISSNLinking'))>0) else '',
    }
    MedlineJournalInfo.getparent().remove(MedlineJournalInfo)

def get_PublicationTypeList():
    PublicationTypeList = get_children_with_tag(Article, 'PublicationTypeList')[0]
    dato['PublicationTypes'] = []
    for PublicationType in get_children_with_tag(PublicationTypeList, 'PublicationType'):
        dato['PublicationTypes'].append(
            {
                'UI': PublicationType.get('UI').strip(),
                'Type': PublicationType.text.strip(),
            }
        )
        PublicationType.getparent().remove(PublicationType)
    PublicationTypeList.getparent().remove(PublicationTypeList)

def get_CommentsCorrectionsList():
    CommentsCorrectionsList = get_children_with_tag(elem, 'CommentsCorrectionsList')
    if(len(CommentsCorrectionsList)>0):
        CommentsCorrectionsList = CommentsCorrectionsList[0]
        dato['references'] = []
        for CommentsCorrections in get_children_with_tag(CommentsCorrectionsList, 'CommentsCorrections'):
            dato['references'].append(
                {
                    'RefType'   : CommentsCorrections.get('RefType').strip(),
                    'RefSource' : get_children_with_tag(CommentsCorrections, 'RefSource')[0].text.strip(),
                    'PMID'      : get_children_with_tag(CommentsCorrections, 'PMID')[0].text.strip() if (len(get_children_with_tag(CommentsCorrections, 'PMID'))>0) else '',
                    'Note'      : get_children_with_tag(CommentsCorrections, 'Note')[0].text.strip() if (len(get_children_with_tag(CommentsCorrections, 'Note'))>0) else '',
                }
            )
            CommentsCorrections.getparent().remove(CommentsCorrections)
        CommentsCorrectionsList.getparent().remove(CommentsCorrectionsList)

def get_Pagination():
    Pagination = get_children_with_tag(Article, 'Pagination')
    if(len(Pagination)>0):
        Pagination = Pagination[0]
        dato['Pagination'] = get_children_with_tag(Pagination, 'MedlinePgn')[0].text.strip()
        Pagination.getparent().remove(Pagination)

def get_ELocationIDs():
    ELocationIDs = get_children_with_tag(Article, 'ELocationID')
    dato['ELocationIDs'] = []
    for ELocationID in ELocationIDs:
        dato['ELocationIDs'].append(
            {
                'EIdType'   : ELocationID.get('EIdType').strip(),
                'EId'       : ELocationID.text.strip(),
            }
        )
        ELocationID.getparent().remove(ELocationID)

def get_Language():
    Language = get_children_with_tag(Article, 'Language')[0]
    dato['Language'] = Language.text.strip()
    Language.getparent().remove(Language)

def get_GrantList():
    GrantList = get_children_with_tag(Article, 'GrantList')
    if(len(GrantList)>0):
        GrantList = GrantList[0]
        dato['Grants'] = []
        for Grant in get_children_with_tag(GrantList, 'Grant'):
            dato['Grants'].append(
                {
                    'GrantID'   : get_children_with_tag(Grant, 'GrantID')[0].text.strip() if(len(get_children_with_tag(Grant, 'GrantID'))>0) else '',
                    'Agency'    : get_children_with_tag(Grant, 'Agency')[0].text.strip(),
                    'Country'   : get_children_with_tag(Grant, 'Country')[0].text.strip(),
                }
            )
            Grant.getparent().remove(Grant)
        GrantList.getparent().remove(GrantList)

def get_KeywordList():
    KeywordList = get_children_with_tag(elem, 'KeywordList')
    if(len(KeywordList)>0):
        KeywordList = KeywordList[0]
        dato['Keywords'] = []
        for Keyword in get_children_with_tag(KeywordList, 'Keyword'):
            if(Keyword.text is not None):
                dato['Keywords'].append(Keyword.text.strip())
            Keyword.getparent().remove(Keyword)
        KeywordList.getparent().remove(KeywordList)

def get_ChemicalList():
    ChemicalList = get_children_with_tag(elem, 'ChemicalList')
    if(len(ChemicalList)>0):
        ChemicalList = ChemicalList[0]
        dato['Chemicals'] = []
        for Chemical in get_children_with_tag(ChemicalList, 'Chemical'):
            dato['Chemicals'].append(
                {
                    'RegistryNumber'    : get_children_with_tag(Chemical, 'RegistryNumber')[0].text.strip(),
                    'NameOfSubstance'   : get_children_with_tag(Chemical, 'NameOfSubstance')[0].text.strip(),
                    'UI'                : get_children_with_tag(Chemical, 'NameOfSubstance')[0].get('UI').strip(),
                }
            )
            Chemical.getparent().remove(Chemical)
        ChemicalList.getparent().remove(ChemicalList)

def get_OtherAbstract():
    OtherAbstract = get_children_with_tag(elem, 'OtherAbstract')
    if(len(OtherAbstract)>0):
        OtherAbstract = OtherAbstract[0]
        dato['OtherAbstract'] = {
            'Type'      : OtherAbstract.get('Type').strip(),
            'Language'  : OtherAbstract.get('Language').strip(),
            'text'      : get_element_lower_text(OtherAbstract).strip(),
        }
        OtherAbstract.getparent().remove(OtherAbstract)

def get_SupplMeshList():
    SupplMeshList = get_children_with_tag(elem, 'SupplMeshList')
    if(len(SupplMeshList)>0):
        SupplMeshList = SupplMeshList[0]
        SupplMeshNames = get_children_with_tag(SupplMeshList, 'SupplMeshName')
        dato['SupplMeshName'] = []
        for SupplMeshName in SupplMeshNames:
            dato['SupplMeshName'].append({
                'text'  : SupplMeshName.text.strip(),
                'Type'  : SupplMeshName.get('Type').strip(),
                'UI'    : SupplMeshName.get('UI').strip(),
            })
            SupplMeshName.getparent().remove(SupplMeshName)
        SupplMeshList.getparent().remove(SupplMeshList)

def get_InvestigatorList():
    InvestigatorList = get_children_with_tag(Article, 'InvestigatorList')
    if(len(InvestigatorList)>0):
        InvestigatorList = InvestigatorList[0]
        Investigators     = get_children_with_tag(InvestigatorList, 'Investigator')
        dato['Investigators'] = []
        for Investigator in Investigators:
            au = {
                    'LastName': get_children_with_tag(Investigator, 'LastName')[0].text.strip() if (len(get_children_with_tag(Investigator, 'LastName'))>0) else '',
                    'ForeName': get_children_with_tag(Investigator, 'ForeName')[0].text.strip() if (len(get_children_with_tag(Investigator, 'ForeName'))>0) else '',
                    'Initials': get_children_with_tag(Investigator, 'Initials')[0].text.strip() if (len(get_children_with_tag(Investigator, 'Initials'))>0) else '',
                    'AffiliationInfo': get_children_with_tag(Investigator, 'AffiliationInfo')[0].text.strip() if (len(get_children_with_tag(Investigator, 'AffiliationInfo'))>0) else '',
                    'CollectiveName': get_children_with_tag(Investigator, 'CollectiveName')[0].text.strip() if (len(get_children_with_tag(Investigator, 'CollectiveName'))>0) else '',
            }
            dato['Investigators'].append(au)
            Investigator.getparent().remove(Investigator)
        InvestigatorList.getparent().remove(InvestigatorList)

def get_NumberOfReferences():
    NumberOfReferences = get_children_with_tag(Article, 'NumberOfReferences')
    if (len(NumberOfReferences) > 0):
        NumberOfReferences = NumberOfReferences[0]
        dato['NumberOfReferences'] = NumberOfReferences.text.strip()
        NumberOfReferences.getparent().remove(NumberOfReferences)

def get_element_lower_text(element, joiner=' '):
    r2  =  create_new_xml_from_element(element)
    return joiner.join(r2.xpath("//text()")).replace('\n',' ')

def get_pmid():
    pmid = get_children_with_tag(elem, 'PMID')[0]
    dato['pmid'] = pmid.text.strip()
    pmid.getparent().remove(pmid)

def get_CitationSubset():
    CitationSubset = get_children_with_tag(elem, 'CitationSubset')
    if(len(CitationSubset)>0):
        CitationSubset = CitationSubset[0]
        dato['CitationSubset'] = CitationSubset.text.strip()
        CitationSubset.getparent().remove(CitationSubset)

def get_DateCreated():
    if(len(get_children_with_tag(elem,'DateCreated'))>0):
        DateCreated             = get_children_with_tag(elem,'DateCreated')[0]
        dato['DateCreated']     = get_children_with_tag(DateCreated,'Day')[0].text.strip() + '/' +get_children_with_tag(DateCreated,'Month')[0].text.strip() + '/' +get_children_with_tag(DateCreated,'Year')[0].text.strip()
        DateCreated.getparent().remove(DateCreated)
    else:
        dato['DateCreated'] = None

def get_DateRevised():
    DateRevised = get_children_with_tag(elem, 'DateRevised')
    if(len(DateRevised)>0):
        DateRevised = DateRevised[0]
        dato['DateRevised']  = get_children_with_tag(DateRevised,'Day')[0].text.strip() + '/' +get_children_with_tag(DateRevised,'Month')[0].text.strip() + '/' +get_children_with_tag(DateRevised,'Year')[0].text.strip()
        DateRevised.getparent().remove(DateRevised)

def get_DateCompleted():
    DateCompleted = get_children_with_tag(elem,'DateCompleted')
    if(len(DateCompleted)>0):
        DateCompleted = DateCompleted[0]
        dato['DateCompleted'] = get_children_with_tag(DateCompleted, 'Day')[0].text.strip() + '/' + \
                                get_children_with_tag(DateCompleted, 'Month')[0].text.strip() + '/' + \
                                get_children_with_tag(DateCompleted, 'Year')[0].text.strip()
        DateCompleted.getparent().remove(DateCompleted)

def get_ArticleTitle():
    ArticleTitle = get_children_with_tag(Article,'ArticleTitle')
    if(len(ArticleTitle)>0 ):
        # print(ArticleTitle)
        ArticleTitle = ArticleTitle[0]
        dato['ArticleTitle'] = get_element_lower_text(ArticleTitle).strip()
        ArticleTitle.getparent().remove(ArticleTitle)
    else:
        dato['ArticleTitle'] = ''

def get_ISOAbbreviation():
    ISOAbbreviation = get_children_with_tag(Journal, 'ISOAbbreviation')
    if(len(ISOAbbreviation)>0):
        ISOAbbreviation = ISOAbbreviation[0]
        dato['ISOAbbreviation'] = ISOAbbreviation.text.strip()
        ISOAbbreviation.getparent().remove(ISOAbbreviation)

def get_ISSN():
    ISSN    = get_children_with_tag(Journal,'ISSN')
    if(len(ISSN)>0):
        ISSN = ISSN[0]
        dato['ISSN'] = ISSN.text.strip()
        ISSN.getparent().remove(ISSN)#

def get_Title():
    Title = get_children_with_tag(Journal,'Title')[0]
    dato['Title'] = Title.text.strip()
    Title.getparent().remove(Title)

def handle_JournalIssue():
    JournalIssue = get_children_with_tag(Journal,'JournalIssue')[0]
    dato['JournalIssue'] = {}
    dato['JournalIssue']['Volume'] = get_children_with_tag(JournalIssue, 'Volume')[0].text.strip() if(len(get_children_with_tag(JournalIssue, 'Volume'))>0) else ''
    issue = get_children_with_tag(JournalIssue, 'Issue')
    if(len(issue)>0):
        dato['JournalIssue']['Issue']  = issue[0].text.strip()
    PubDate = get_children_with_tag(JournalIssue, 'PubDate')[0]
    #
    date = get_children_with_tag(PubDate, 'Year')[0].text.strip() if (len(get_children_with_tag(PubDate, 'Year'))>0) else ''
    #
    m = get_children_with_tag(PubDate, 'Month')
    if(len(m)>0):
        mo = m[0].text.strip()
        if(str.isdigit(mo)):
            date = mo +'/'+ date
        else:
            date = mo +' '+ date
    #
    d = get_children_with_tag(PubDate, 'Day')
    if(len(d)>0):
        date = d[0].text.strip()+'/'+date
    dato['JournalIssue']['PubDate'] = date
    dato['JournalIssue']['CitedMedium'] = JournalIssue.get('CitedMedium')
    JournalIssue.getparent().remove(JournalIssue)

def create_elk_basic_dato(dato):
    elk_dato = {}
    keys_to_hold = [
        'Title', 'ArticleTitle',
        'Keywords', 'CitationSubset', 'ISOAbbreviation',
        'Pagination', 'ISSN', 'pmid', 'Language', 'Authors', 'MedlineJournalInfo',
        'SupplMeshName', 'PublicationTypes',
        'Chemicals', 'OtherIDs', 'references',
    ]
    #
    for k in keys_to_hold:
        elk_dato[k] = dato[k] if(k in dato.keys()) else None
    #
    keys_to_hold2 = ['DateCompleted', 'DateCreated', 'ArticleDate', 'DateRevised', ]
    for k in keys_to_hold2:
        elk_dato[k] = parser.parse(dato[k]) if(k in dato.keys()) else None
    #
    if('JournalIssue' in dato.keys()):
        if(type(dato['JournalIssue']) is str):
            dato['JournalIssue']['PubDate'] = parser.parse(dato['JournalIssue']['PubDate'])
            elk_dato['JournalIssue'] = dato['JournalIssue']
    else:
        elk_dato['JournalIssue'] = None
    #
    if ('MeshHeadings' in dato.keys()):
        elk_dato['MeshHeadings']= []
        for mh in dato['MeshHeadings']:
            for k in mh.keys():
                # try:
                elk_dato['MeshHeadings'].append({
                    "UI"    : mh[k]['UI'],
                    "Name"  : k,
                    "Type"  : mh[k]['Type'],
                    "text"  : mh[k]['text'],
                })
                # except:
                #     pprint(mh)
                #     exit(0)
    #
    return elk_dato

def fix_all_elk_data(dato):
    elk_data = []
    #
    if 'AbstractText' in dato.keys():
        for item in dato['AbstractText']:
            bd = create_elk_basic_dato(dato)
            bd['AbstractText']          = item['text']
            bd['AbstractLabel']         = item['Label']
            bd['AbstractNlmCategory']   = item['NlmCategory']
            bd['AbstractLanguage']      = item['AbstractLanguage']   if ('AbstractLanguage' in item.keys()) else None
            bd['AbstractType']          = item['AbstractType']       if ('AbstractType' in item.keys()) else None
            elk_data.append(bd)
    #
    if 'OtherAbstract' in dato.keys():
        item = dato['OtherAbstract']
        bd = create_elk_basic_dato(dato)
        bd['AbstractText']              = item['text']
        bd['AbstractLabel']             = item['Label']             if ('Label' in item.keys()) else None
        bd['AbstractNlmCategory']       = item['NlmCategory']       if ('NlmCategory' in item.keys()) else None
        bd['AbstractLanguage']          = item['AbstractLanguage']  if ('AbstractLanguage' in item.keys()) else None
        bd['AbstractType']              = item['AbstractType']      if ('AbstractType' in item.keys()) else None
        elk_data.append(bd)
    #
    return elk_data

def fix_elk_dato(dato):
    try:
        dato['AbstractText'] = '\n'.join([t['text'] for t in dato['AbstractText']]).strip()
    except KeyError:
        dato['AbstractText'] = ''
    # if ('OtherAbstract' in dato.keys()):
    #     dato['AbstractText'] += '\n' + dato['OtherAbstract']['text'].strip()
    #     del (dato['OtherAbstract'])
    dato['AbstractText'] = dato['AbstractText'].strip()
    if('MeshHeadings' in dato):
        t = []
        for item in dato['MeshHeadings']:
            for v in item.values():
                t.append({
                    "UI"    : v['UI'],
                    "name"  : v['text']
                })
        dato['MeshHeadings'] = t
    if('SupplMeshName' in dato):
        t = []
        for v in dato['SupplMeshName']:
            t.append({
                "UI"    : v['UI'],
                "Type"  : v['Type'],
                "name"  : v['text']
            })
        dato['SupplMeshName'] = t
        #
    if(dato['DateCreated'] is None and 'DateCompleted' in dato):
        dato['DateCreated'] = dato['DateCompleted']
    if(dato['DateCreated'] is None and 'DateRevised' in dato):
        dato['DateCreated'] = dato['DateRevised']
    try:
        del(dato['DateCompleted'])
    except:
        pass
    try:
        del(dato['DateRevised'])
    except:
        pass
    return dato

def create_an_action(tw):
    tw['_op_type']= 'index'
    tw['_index']= index
    tw['_type']= doc_type
    return tw

def send_to_elk(actions):
    flag = True
    while (flag):
        try:
            result = bulk(es, iter(actions))
            pprint(result)
            flag = False
        except Exception as e:
            print(e)
            if ('ConnectionTimeout' in str(e)):
                print('Retrying')
            else:
                flag = False

def clean_alittle(content):
    content     = content.replace('<sub>','_ssub_')
    content     = content.replace('</sub>','_esub_')
    content     = content.replace('<sup>','_ssup_')
    content     = content.replace('</sup>','_esup_')
    return content

diri    = '/media/dpappas/Maxtor/Pubmed_abstract_baselines/'
fs      = [ diri+f for f in os.listdir(diri) if f.endswith('.xml.gz') ]
fs.sort(reverse=True)
# random.shuffle(fs)

b_size  = 500
actions = []

fc = 0
for file_gz in fs[300:400]:
    fc += 1
    infile      = gzip.open(file_gz)
    content     = infile.read()
    content     = clean_alittle(content)
    children    = etree.fromstring(content).getchildren()
    ch_counter  = 0
    for ch_tree in children:
        ch_counter += 1
        for elem in ch_tree.iter(tag='MedlineCitation'):
            elem    = etree.fromstring(etree.tostring(elem))
            Article = get_children_with_tag(elem, 'Article')[0]
            Journal = get_children_with_tag(Article, 'Journal')[0]
            dato = {}
            # print(etree.tostring(elem))
            try:
                get_pmid()
                # get_NumberOfReferences()
                # get_InvestigatorList()
                get_SupplMeshList()
                get_ChemicalList()
                get_MeshHeadings()
                # get_PersonalNameSubjectList()
                # get_OtherIDs()
                get_KeywordList()
                # get_CommentsCorrectionsList()
                # get_GrantList()
                # get_ELocationIDs()
                # get_Language()
                # get_PublicationTypeList()
                # get_MedlineJournalInfo()
                # get_Pagination()
                # get_CitationSubset()
                get_DateCreated()
                get_DateRevised()
                get_DateCompleted()
                get_ArticleTitle()
                # get_Authors()
                # get_ISOAbbreviation()
                # get_ISSN()
                get_Title()
                # handle_JournalIssue()
                get_ArticleDate()
                get_Abstract()
                # get_OtherAbstract()
            except:
                print etree.tostring(elem, pretty_print=True)
                traceback.print_exc()
                tb = traceback.format_exc()
                print tb
            # pprint(dato)
            dato = fix_elk_dato(dato)
            # pprint(dato)
            if (not abs_found(dato['pmid'])):
                temp = create_an_action(dato)
                actions.append(temp)
            else:
                print 'found pmid {}'.format(dato['pmid'])
            if (len(actions) >= b_size):
                send_to_elk(actions)
                actions = []
        print('finished {} of {} trees. {} of {} files. found items up_to_now:{}'.format(ch_counter, len(children), fc, len(fs), len(actions)))

if(len(actions) > 0):
    send_to_elk(actions)
    actions = []



