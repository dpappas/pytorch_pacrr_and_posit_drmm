#!/usr/bin/env python

import re
import os
import gzip
import traceback
import cPickle as pickle
from pprint import pprint
from lxml import etree
from dateutil import parser
import random
import mechanize
import cookielib
from bs4 import BeautifulSoup
import cPickle as pickle
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import urllib2  # the lib that handles the url stuff
import json

def get_br():
    # Browser
    br = mechanize.Browser()
    # Cookie Jar
    cj = cookielib.LWPCookieJar()
    br.set_cookiejar(cj)
    # Browser options
    br.set_handle_equiv(True)
    br.set_handle_gzip(True)
    br.set_handle_redirect(True)
    br.set_handle_referer(True)
    br.set_handle_robots(False)
    # Follows refresh 0 but not hangs on refresh > 0
    br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)
    # br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1'), ('Accept', '*/*')]
    br.addheaders = [('User-agent', "Mozilla/5.0 (X11; Linux x86_64; rv:44.0) Gecko/20100101 Firefox/44.0"),
                     ('Accept', '*/*')]
    return br

def get_soup(link):
    br = get_br()
    html = br.open(link).read()
    # link = br.geturl()
    # link = '/'.join(link.split('/')[:-1])+'/'
    # soup = BeautifulSoup(html)
    soup = BeautifulSoup(html, "lxml")
    return soup

def create_new_xml_from_element(element):
    return etree.fromstring(etree.tostring(element))

def get_children_with_tag(elem,tag):
    return [ x for x in elem.getchildren() if(x.tag == tag) ]

def get_OtherIDs(dato, root):
    OtherIDs = get_children_with_tag(root, 'PubmedData')[0]
    OtherIDs = get_children_with_tag(OtherIDs,'ArticleIdList')[0]
    OtherIDs = get_children_with_tag(OtherIDs,'ArticleId')
    dato['OtherIDs'] = [
        {
            'Source'    :id.get('IdType').strip(),
            'id'        :id.text.strip(),
        } for id in OtherIDs
    ]
    for OtherID in OtherIDs:
        OtherID.getparent().remove(OtherID)

def get_MeshHeadings(dato, elem):
    MeshHeadingLists        = get_children_with_tag(elem, 'MeshHeadingList')
    dato['MeshHeadings']    = []
    for MeshHeadingList in MeshHeadingLists:
        MeshHeadings = get_children_with_tag(MeshHeadingList, 'MeshHeading')
        for MeshHeading in MeshHeadings:
            mh = []
            for item in MeshHeading.getchildren():
                mh.append({
                    'Label':        item.tag.strip(),
                    'text':         item.text.strip(),
                    'UI':           item.get('UI').strip(),
                    'MajorTopicYN': item.get('MajorTopicYN').strip(),
                    'Type':         item.get('Type').strip() if (item.get('Type') is not None) else '',
                })
            dato['MeshHeadings'].append(mh)
            MeshHeading.getparent().remove(MeshHeading)
        MeshHeadingList.getparent().remove(MeshHeadingList)

def get_Authors(dato, Article):
    AuthorList = get_children_with_tag(Article, 'AuthorList')
    if(len(AuthorList)>0):
        AuthorList = AuthorList[0]
        Authors     = get_children_with_tag(AuthorList, 'Author')
        dato['Authors'] = []
        for Author in Authors:
            au = {
                    'LastName':         get_children_with_tag(Author, 'LastName')[0].text.strip() if (len(get_children_with_tag(Author, 'LastName'))>0) else '',
                    'ForeName':         get_children_with_tag(Author, 'ForeName')[0].text.strip() if (len(get_children_with_tag(Author, 'ForeName'))>0) else '',
                    'Initials':         get_children_with_tag(Author, 'Initials')[0].text.strip() if (len(get_children_with_tag(Author, 'Initials'))>0) else '',
                    'AffiliationInfo':  [
                        get_element_lower_text(af_inf).strip()
                        for af_inf in get_children_with_tag(Author, 'AffiliationInfo')
                    ],
                    'CollectiveName':   get_children_with_tag(Author, 'CollectiveName')[0].text.strip() if (len(get_children_with_tag(Author, 'CollectiveName'))>0) else '',
            }
            dato['Authors'].append(au)
            Author.getparent().remove(Author)
        AuthorList.getparent().remove(AuthorList)

def get_PersonalNameSubjectList(dato, elem):
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

def get_Abstract(dato, Article):
    text = ''
    for Abstract in get_children_with_tag(Article, 'Abstract'):
        for item in Abstract.getchildren():
            if(item.tag.strip() != 'CopyrightInformation'):
                label = item.get('Label')
                if(label is not None):
                    text += '\n'+label.strip()+'\n\n'
                text += get_element_lower_text(item).strip()+'\n'
            item.getparent().remove(item)
    dato['AbstractText'] = text.strip()

def get_ArticleDate(dato, Article):
    ArticleDate = get_children_with_tag(Article, 'ArticleDate')
    if(len(ArticleDate)>0):
        ArticleDate = ArticleDate[0]
        dato['ArticleDate']  = get_children_with_tag(ArticleDate,'Day')[0].text.strip() + '/' \
                               +get_children_with_tag(ArticleDate,'Month')[0].text.strip() + '/' \
                               +get_children_with_tag(ArticleDate,'Year')[0].text.strip()
        ArticleDate.getparent().remove(ArticleDate)

def get_PublicationTypeList(dato, Article):
    try:
        dato['PublicationTypes'] = []
        for PublicationTypeList in get_children_with_tag(Article, 'PublicationTypeList'):
            for PublicationType in get_children_with_tag(PublicationTypeList, 'PublicationType'):
                dato['PublicationTypes'].append(
                    {
                        'UI':   PublicationType.get('UI').strip(),
                        'Type': PublicationType.text.strip(),
                    }
                )
                PublicationType.getparent().remove(PublicationType)
    except:
        dato['PublicationTypes'] = []

def get_CommentsCorrectionsList(dato, elem):
    CommentsCorrectionsList = get_children_with_tag(elem, 'CommentsCorrectionsList')
    if(len(CommentsCorrectionsList)>0):
        CommentsCorrectionsList = CommentsCorrectionsList[0]
        dato['references'] = []
        for CommentsCorrections in get_children_with_tag(CommentsCorrectionsList, 'CommentsCorrections'):
            dato['references'].append(
                {
                    'RefType'   : CommentsCorrections.get('RefType').strip(),
                    'RefSource' : get_children_with_tag(CommentsCorrections, 'RefSource')[0].text.strip() if (len(get_children_with_tag(CommentsCorrections, 'RefSource'))>0 and get_children_with_tag(CommentsCorrections, 'RefSource')[0].text is not None) else '',
                    'PMID'      : get_children_with_tag(CommentsCorrections, 'PMID')[0].text.strip() if (len(get_children_with_tag(CommentsCorrections, 'PMID'))>0 and get_children_with_tag(CommentsCorrections, 'PMID')[0].text is not None) else '',
                    'Note'      : get_children_with_tag(CommentsCorrections, 'Note')[0].text.strip() if (len(get_children_with_tag(CommentsCorrections, 'Note'))>0 and get_children_with_tag(CommentsCorrections, 'Note')[0].text is not None) else '',
                }
            )
            CommentsCorrections.getparent().remove(CommentsCorrections)
        CommentsCorrectionsList.getparent().remove(CommentsCorrectionsList)

def get_Pagination(dato, Article):
    Pagination = get_children_with_tag(Article, 'Pagination')
    if(len(Pagination)>0):
        try:
            Pagination = Pagination[0]
            dato['Pagination'] = get_children_with_tag(Pagination, 'MedlinePgn')[0].text.strip()
            Pagination.getparent().remove(Pagination)
        except:
            dato['Pagination'] = ''

def get_Language(dato, Article):
    try:
        Language = get_children_with_tag(Article, 'Language')[0]
        dato['Language'] = Language.text.strip()
        Language.getparent().remove(Language)
    except:
        dato['Language'] = ''

def get_GrantList(dato, Article):
    dato['Grants'] = []
    GrantLists = get_children_with_tag(Article, 'GrantList')
    for GrantList in GrantLists:
        Grants = get_children_with_tag(GrantList, 'Grant')
        for Grant in Grants:
            dato['Grants'].append(
                {
                    'GrantID': get_children_with_tag(Grant, 'GrantID')[0].text.strip() if (len(get_children_with_tag(Grant, 'GrantID')) > 0) else '',
                    'Agency':  get_children_with_tag(Grant, 'Agency')[0].text.strip()  if (len(get_children_with_tag(Grant, 'Agency')) > 0) else '',
                    'Country': get_children_with_tag(Grant, 'Country')[0].text.strip() if (
                            len(get_children_with_tag(Grant, 'Country')) > 0
                            and
                            get_children_with_tag(Grant, 'Country')[0].text is not None
                    ) else '',
                }
            )
            Grant.getparent().remove(Grant)
        GrantList.getparent().remove(GrantList)

def get_KeywordList(dato, elem):
    KeywordList = get_children_with_tag(elem, 'KeywordList')
    if(len(KeywordList)>0):
        KeywordList = KeywordList[0]
        dato['Keywords'] = []
        for Keyword in get_children_with_tag(KeywordList, 'Keyword'):
            if(Keyword.text is not None):
                dato['Keywords'].append(Keyword.text.strip())
            Keyword.getparent().remove(Keyword)
        KeywordList.getparent().remove(KeywordList)

def get_ChemicalList(dato, elem):
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

def get_OtherAbstract(dato, Article):
    OtherAbstract = get_children_with_tag(Article, 'OtherAbstract')
    if(len(OtherAbstract)>0):
        OtherAbstract = OtherAbstract[0]
        dato['OtherAbstract'] = {
            'Type'      : OtherAbstract.get('Type').strip(),
            'Language'  : OtherAbstract.get('Language').strip(),
            'text'      : get_element_lower_text(OtherAbstract).strip(),
        }
        OtherAbstract.getparent().remove(OtherAbstract)

def get_SupplMeshList(dato, elem):
    dato['SupplMeshList']    = []
    MeshHeadingLists        = get_children_with_tag(elem, 'SupplMeshList')
    for MeshHeadingList in MeshHeadingLists:
        MeshHeadings = get_children_with_tag(MeshHeadingList, 'SupplMeshName')
        for MeshHeading in MeshHeadings:
            dato['SupplMeshList'].append({
                'text'  : MeshHeading.text.strip(),
                'Type'  : MeshHeading.get('Type').strip(),
                'UI'    : MeshHeading.get('UI').strip(),
            })
            MeshHeading.getparent().remove(MeshHeading)
        MeshHeadingList.getparent().remove(MeshHeadingList)

def get_InvestigatorList(dato, Article):
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
                    'AffiliationInfo': [
                        get_element_lower_text(af_inf).strip()
                        for af_inf in get_children_with_tag(Investigator, 'AffiliationInfo')
                    ],
                    'CollectiveName': get_children_with_tag(Investigator, 'CollectiveName')[0].text.strip() if (len(get_children_with_tag(Investigator, 'CollectiveName'))>0) else '',
            }
            dato['Investigators'].append(au)
            Investigator.getparent().remove(Investigator)
        InvestigatorList.getparent().remove(InvestigatorList)

def get_NumberOfReferences(dato, Article):
    NumberOfReferences = get_children_with_tag(Article, 'NumberOfReferences')
    if (len(NumberOfReferences) > 0):
        NumberOfReferences = NumberOfReferences[0]
        dato['NumberOfReferences'] = NumberOfReferences.text.strip()
        NumberOfReferences.getparent().remove(NumberOfReferences)

def get_element_lower_text(element, joiner=' '):
    r2  =  create_new_xml_from_element(element)
    return joiner.join(r2.xpath("//text()")).replace('\n',' ')

def get_pmid(dato, elem):
    pmid = get_children_with_tag(elem, 'PMID')[0]
    dato['pmid'] = pmid.text.strip()
    pmid.getparent().remove(pmid)

def get_CitationSubset(dato, elem):
    CitationSubset = get_children_with_tag(elem, 'CitationSubset')
    if(len(CitationSubset)>0):
        CitationSubset = CitationSubset[0]
        dato['CitationSubset'] = CitationSubset.text.strip()
        CitationSubset.getparent().remove(CitationSubset)

def get_DateCreated(dato, elem):
    ttt = get_children_with_tag(elem,'DateCreated')
    if(len(ttt)>0):
        DateCreated   = ttt[0]
        dato['DateCreated']  = get_children_with_tag(DateCreated,'Day')[0].text.strip() + '/' +get_children_with_tag(DateCreated,'Month')[0].text.strip() + '/' +get_children_with_tag(DateCreated,'Year')[0].text.strip()
        DateCreated.getparent().remove(DateCreated)
    return None

def get_DateRevised(dato, elem):
    DateRevised = get_children_with_tag(elem, 'DateRevised')
    if(len(DateRevised)>0):
        DateRevised = DateRevised[0]
        dato['DateRevised']  = get_children_with_tag(DateRevised,'Day')[0].text.strip() + '/' +get_children_with_tag(DateRevised,'Month')[0].text.strip() + '/' +get_children_with_tag(DateRevised,'Year')[0].text.strip()
        DateRevised.getparent().remove(DateRevised)

def get_DateCompleted(dato, elem):
    DateCompleted = get_children_with_tag(elem,'DateCompleted')
    if(len(DateCompleted)>0):
        DateCompleted = DateCompleted[0]
        dato['DateCompleted'] = get_children_with_tag(DateCompleted, 'Day')[0].text.strip() + '/' + \
                                get_children_with_tag(DateCompleted, 'Month')[0].text.strip() + '/' + \
                                get_children_with_tag(DateCompleted, 'Year')[0].text.strip()
        DateCompleted.getparent().remove(DateCompleted)

def get_ArticleTitle(dato, Article):
    if(len(get_children_with_tag(Article,'ArticleTitle'))>0):
        ArticleTitle = get_children_with_tag(Article,'ArticleTitle')[0]
        try:
            dato['ArticleTitle'] = ArticleTitle.text.strip()
        except:
            dato['ArticleTitle'] = ''
        ArticleTitle.getparent().remove(ArticleTitle)

def do_for_one_pmid(pmid):
    dato = {}
    link = 'https://www.ncbi.nlm.nih.gov/pubmed/?term={}&report=xml&format=json'.format(pmid)
    data = urllib2.urlopen(link).read()
    data = data.replace('&lt;', '<')
    data = data.replace('&gt;', '>')
    if('PubmedArticle' in data):
        etree_root  = etree.fromstring(data)
        root        = get_children_with_tag(etree_root, 'PubmedArticle')[0]
        elem        = get_children_with_tag(root, 'MedlineCitation')[0]
        Article     = get_children_with_tag(elem, 'Article')[0]
        try:
            get_pmid(dato, elem)
            # get_CitationSubset(dato, elem)
            get_DateCreated(dato, elem)
            get_DateRevised(dato, elem)
            # get_NumberOfReferences(dato, Article)
            # get_InvestigatorList(dato, Article)
            get_SupplMeshList(dato, elem)
            get_ChemicalList(dato, elem)
            get_MeshHeadings(dato, elem)
            get_PersonalNameSubjectList(dato, elem)
            get_OtherIDs(dato, root)
            get_KeywordList(dato, elem)
            # get_CommentsCorrectionsList(dato, elem)
            get_GrantList(dato, Article)
            get_Language(dato, Article)
            # get_PublicationTypeList(dato, Article)
            # get_Pagination(dato, Article)
            get_DateCompleted(dato, elem)
            get_ArticleTitle(dato, Article)
            # get_Authors(dato, Article)
            get_ArticleDate(dato, Article)
            get_Abstract(dato, Article)
            get_OtherAbstract(dato, Article)
        except:
            print etree.tostring(elem, pretty_print=True)
            traceback.print_exc()
            tb = traceback.format_exc()
            print tb
    else:
        pass
    return dato

def load_all_data(dataloc):
    print('loading pickle data')
    #
    with open(dataloc+'BioASQ-trainingDataset6b.json', 'r') as f:
        bioasq6_data = json.load(f)
        bioasq6_data = dict((q['id'], q) for q in bioasq6_data['questions'])
    with open(dataloc + 'bioasq_bm25_top100.test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.dev.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    with open(dataloc + 'bioasq_bm25_top100.train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    print('loading words')
    #
    return test_data, dev_data, train_data, bioasq6_data

def get_all_ids_from_data(test_data, dev_data, train_data, bioasq6_data):
    all_ids = []
    for quer in train_data['queries']+dev_data['queries']+test_data['queries']:
        all_ids.extend([rd['doc_id'] for rd in quer['retrieved_documents']])
    for val in bioasq6_data.values():
        all_ids.extend([d.split('/')[-1] for d in val['documents']])
        if('snippets' in val):
            all_ids.extend([sn['document'].split('/')[-1] for sn in val['snippets']])
    all_ids = list(set(all_ids))
    return all_ids

w2v_bin_path    = '/home/dpappas/for_ryan/fordp/pubmed2018_w2v_30D.bin'
idf_pickle_path = '/home/dpappas/for_ryan/fordp/idf.pkl'
dataloc         = '/home/dpappas/for_ryan/'

(test_data, dev_data, train_data, bioasq6_data) = load_all_data(dataloc=dataloc)

all_ids = get_all_ids_from_data(test_data, dev_data, train_data, bioasq6_data)

odir = './downloaded/'
if(not os.path.exists(odir)):
    os.makedirs(odir)

random.shuffle(all_ids)
for pmid in tqdm(all_ids):
    opath   = os.path.join(odir,'{}.json'.format(pmid))
    if(not os.path.exists(opath)):
        dato    = do_for_one_pmid(pmid)
        with open(opath, 'w') as f:
            f.write(json.dumps(dato, indent=4, sort_keys=True))
            f.close()



