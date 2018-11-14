
import gc
gc.collect()

import os
import numpy as np
from pprint import pprint

def init_dic():
    return {
            'dev': {
                'map_doc'    : 0.,
                'map_doc_bioasq'    : 0.,
                'f1_snip'           : 0.,
                'map_snip'          : 0.,
                'gmap_snip'         : 0.,
                'known_f1_snip'     : 0.,
                'known_map_snip'    : 0.,
                'known_gmap_snip'   : 0.,
            },
            'test': {
                'map_doc'    : 0.,
                'map_doc_bioasq'    : 0.,
                'f1_snip'           : 0.,
                'map_snip'          : 0.,
                'gmap_snip'         : 0.,
                'known_f1_snip'     : 0.,
                'known_map_snip'    : 0.,
                'known_gmap_snip'   : 0.,
            },
            'epoch'                 : 0.,
            # 'time'                  : 0.,
        }

def assign_data(data, res1, res2, res3, the_line):
    res1['test']['map_doc']         = float(the_line.split(':')[-1].strip())
    res1['dev']['map_doc']          = float(the_line.split('epoch_dev_map:')[1].split()[0].strip())
    res1['epoch']                   = int(the_line.split('epoch:')[1].split()[0])
    res2['test']['map_doc']         = float(the_line.split(':')[-1].strip())
    res2['dev']['map_doc']          = float(the_line.split('epoch_dev_map:')[1].split()[0].strip())
    res2['epoch']                   = int(the_line.split('epoch:')[1].split()[0])
    res3['test']['map_doc']         = float(the_line.split(':')[-1].strip())
    res3['dev']['map_doc']          = float(the_line.split('epoch_dev_map:')[1].split()[0].strip())
    res3['epoch']                   = int(the_line.split('epoch:')[1].split()[0])
    #
    res1['dev']['known_map_doc_bioasq'] = data[0]
    res1['dev']['known_f1_snip']        = data[1]
    res1['dev']['known_map_snip']       = data[2]
    res1['dev']['known_gmap_snip']      = data[3]
    res1['dev']['map_doc_bioasq']       = data[4]
    res1['dev']['f1_snip']              = data[5]
    res1['dev']['map_snip']             = data[6]
    res1['dev']['gmap_snip']            = data[7]
    #
    res2['dev']['known_map_doc_bioasq'] = data[8]
    res2['dev']['known_f1_snip']        = data[9]
    res2['dev']['known_map_snip']       = data[10]
    res2['dev']['known_gmap_snip']      = data[11]
    res2['dev']['map_doc_bioasq']       = data[12]
    res2['dev']['f1_snip']              = data[13]
    res2['dev']['map_snip']             = data[14]
    res2['dev']['gmap_snip']            = data[15]
    #
    res3['dev']['known_map_doc_bioasq'] = data[16]
    res3['dev']['known_f1_snip']        = data[17]
    res3['dev']['known_map_snip']       = data[18]
    res3['dev']['known_gmap_snip']      = data[19]
    res3['dev']['map_doc_bioasq']       = data[20]
    res3['dev']['f1_snip']              = data[21]
    res3['dev']['map_snip']             = data[22]
    res3['dev']['gmap_snip']            = data[23]
    #
    res1['test']['known_map_doc_bioasq']    = data[24]
    res1['test']['known_f1_snip']           = data[25]
    res1['test']['known_map_snip']          = data[26]
    res1['test']['known_gmap_snip']         = data[27]
    res1['test']['map_doc_bioasq']          = data[28]
    res1['test']['f1_snip']                 = data[29]
    res1['test']['map_snip']                = data[30]
    res1['test']['gmap_snip']               = data[31]
    #
    res2['test']['known_map_doc_bioasq']    = data[32]
    res2['test']['known_f1_snip']           = data[33]
    res2['test']['known_map_snip']          = data[34]
    res2['test']['known_gmap_snip']         = data[35]
    res2['test']['map_doc_bioasq']          = data[36]
    res2['test']['f1_snip']                 = data[37]
    res2['test']['map_snip']                = data[38]
    res2['test']['gmap_snip']               = data[39]
    #
    res3['test']['known_map_doc_bioasq']    = data[40]
    res3['test']['known_f1_snip']           = data[41]
    res3['test']['known_map_snip']          = data[42]
    res3['test']['known_gmap_snip']         = data[43]
    res3['test']['map_doc_bioasq']          = data[44]
    res3['test']['f1_snip']                 = data[45]
    res3['test']['map_snip']                = data[46]
    res3['test']['gmap_snip']               = data[47]
    return res1, res2, res3

def print_the_results(all1, all2, all3):
    print ''
    print diri
    print ''
    print 'V1'
    print '\n'.join('\t'.join(str(e) for e in t) for t in all1)
    print '\t'.join(str(e) for e in np.average(np.array(all1, dtype='float'), axis=0).tolist())
    print '\t'.join(str(e) for e in np.max(np.array(all1, dtype='float'), axis=0).tolist())
    print '\t'.join(str(e) for e in np.min(np.array(all1, dtype='float'), axis=0).tolist())
    print ''
    print 'V2'
    print '\n'.join('\t'.join(str(e) for e in t) for t in all2)
    print '\t'.join(str(e) for e in np.average(np.array(all2, dtype='float'), axis=0).tolist())
    print '\t'.join(str(e) for e in np.max(np.array(all2, dtype='float'), axis=0).tolist())
    print '\t'.join(str(e) for e in np.min(np.array(all2, dtype='float'), axis=0).tolist())
    print ''
    print 'V3'
    print '\n'.join('\t'.join(str(e) for e in t) for t in all3)
    print '\t'.join(str(e) for e in np.average(np.array(all3, dtype='float'), axis=0).tolist())
    print '\t'.join(str(e) for e in np.max(np.array(all3, dtype='float'), axis=0).tolist())
    print '\t'.join(str(e) for e in np.min(np.array(all3, dtype='float'), axis=0).tolist())
    print ''

def do_three_losses():
    all1 = []
    all2 = []
    all3 = []
    for i in range(5):
        fpath = diri.format(i)
        if(os.path.exists(fpath)):
            res3 = init_dic()
            res2 = init_dic()
            res1 = init_dic()
            with open(fpath) as f:
                lines = f.readlines()
                for l in range(len(lines)):
                    if('v3 test known MAP documents' in lines[l]):
                        data = lines[l-40:l+8]
                        data                            = [float(t.strip().split()[-1]) for t in data]
                        #
                        # pprint(data)
                        res1, res2, res3                = assign_data(data, res1, res2, res3, lines[l+8])
                        #
                all1.append(
                    (
                        res1['test']['map_doc'],
                        res1['test']['map_doc_bioasq'],
                        res1['test']['f1_snip'],
                        res1['test']['map_snip'],
                        res1['test']['gmap_snip'],
                        res1['test']['known_f1_snip'],
                        res1['test']['known_map_snip'],
                        res1['test']['known_gmap_snip'],
                        res1['dev']['map_doc'],
                        res1['dev']['map_doc_bioasq'],
                        res1['dev']['f1_snip'],
                        res1['dev']['map_snip'],
                        res1['dev']['gmap_snip'],
                        res1['dev']['known_f1_snip'],
                        res1['dev']['known_map_snip'],
                        res1['dev']['known_gmap_snip'],
                        res1['epoch'],
                    )
                )
                all2.append(
                    (
                        res2['test']['map_doc'],
                        res2['test']['map_doc_bioasq'],
                        res2['test']['f1_snip'],
                        res2['test']['map_snip'],
                        res2['test']['gmap_snip'],
                        res2['test']['known_f1_snip'],
                        res2['test']['known_map_snip'],
                        res2['test']['known_gmap_snip'],
                        res2['dev']['map_doc'],
                        res2['dev']['map_doc_bioasq'],
                        res2['dev']['f1_snip'],
                        res2['dev']['map_snip'],
                        res2['dev']['gmap_snip'],
                        res2['dev']['known_f1_snip'],
                        res2['dev']['known_map_snip'],
                        res2['dev']['known_gmap_snip'],
                        res2['epoch'],
                    )
                )
                all3.append(
                    (
                        res3['test']['map_doc'],
                        res3['test']['map_doc_bioasq'],
                        res3['test']['f1_snip'],
                        res3['test']['map_snip'],
                        res3['test']['gmap_snip'],
                        res3['test']['known_f1_snip'],
                        res3['test']['known_map_snip'],
                        res3['test']['known_gmap_snip'],
                        res3['dev']['map_doc'],
                        res3['dev']['map_doc_bioasq'],
                        res3['dev']['f1_snip'],
                        res3['dev']['map_snip'],
                        res3['dev']['gmap_snip'],
                        res3['dev']['known_f1_snip'],
                        res3['dev']['known_map_snip'],
                        res3['dev']['known_gmap_snip'],
                        res3['epoch'],
                    )
                )
    print_the_results(all1, all2, all3)

def do_one_loss():
    tests   = []
    devs    = []
    overall = []
    for i in range(5):
        fpath = diri.format(i)
        if (os.path.exists(fpath)):
            res1 = init_dic()
            with open(fpath) as f:
                lines = f.readlines()
                for l in range(len(lines)):
                    if ('test known MAP documents' in lines[l]):
                        data = lines[l-8: l + 8]
                        # pprint(data)
                        res1['test']['map_doc'] = float(lines[l + 8].split(':')[-1].strip())
                        res1['dev']['map_doc'] = float(lines[l + 8].split('epoch_dev_map:')[1].split()[0].strip())
                        res1['epoch'] = int(lines[l + 8].split('epoch:')[1].split()[0])
                        data = [float(t.strip().split()[-1]) for t in data]
                        # pprint(data)
                        # print(len(data))
                        res1['dev']['known_map_doc_bioasq']     = data[0]
                        res1['dev']['known_f1_snip']            = data[1]
                        res1['dev']['known_map_snip']           = data[2]
                        res1['dev']['known_gmap_snip']          = data[3]
                        res1['dev']['map_doc_bioasq']           = data[4]
                        res1['dev']['f1_snip']                  = data[5]
                        res1['dev']['map_snip']                 = data[6]
                        res1['dev']['gmap_snip']                = data[7]
                        #
                        res1['test']['known_map_doc_bioasq']    = data[8]
                        res1['test']['known_f1_snip']           = data[9]
                        res1['test']['known_map_snip']          = data[10]
                        res1['test']['known_gmap_snip']         = data[11]
                        res1['test']['map_doc_bioasq']          = data[12]
                        res1['test']['f1_snip']                 = data[13]
                        res1['test']['map_snip']                = data[14]
                        res1['test']['gmap_snip']               = data[15]
                        #
                        #
                overall.append(
                    (
                        res1['test']['map_doc'], res1['test']['map_doc_bioasq'], res1['test']['f1_snip'],
                        res1['test']['map_snip'], res1['test']['gmap_snip'], res1['test']['known_f1_snip'],
                        res1['test']['known_map_snip'], res1['test']['known_gmap_snip'], res1['dev']['map_doc'],
                        res1['dev']['map_doc_bioasq'], res1['dev']['f1_snip'], res1['dev']['map_snip'],
                        res1['dev']['gmap_snip'], res1['dev']['known_f1_snip'], res1['dev']['known_map_snip'],
                        res1['dev']['known_gmap_snip'], res1['epoch']
                    )
                )
                tests.append('{}\t{}\t{}\t{}\t{}\t{}'.format(res1['test']['f1_snip'], res1['test']['map_snip'],
                                                             res1['test']['gmap_snip'], res1['test']['known_f1_snip'],
                                                             res1['test']['known_map_snip'],
                                                             res1['test']['known_gmap_snip']))
                devs.append('{}\t{}\t{}\t{}\t{}\t{}'.format(res1['dev']['f1_snip'], res1['dev']['map_snip'],
                                                            res1['dev']['gmap_snip'], res1['dev']['known_f1_snip'],
                                                            res1['dev']['known_map_snip'],
                                                            res1['dev']['known_gmap_snip']))
    print '\n'.join(
        '\t'.join(str(e) for e in t)
        for t in overall
    )
    print '\t'.join(str(e) for e in np.average(np.array(overall, dtype='float'), axis=0).tolist())
    print '\t'.join(str(e) for e in np.max(np.array(overall, dtype='float'), axis=0).tolist())
    print '\t'.join(str(e) for e in np.min(np.array(overall, dtype='float'), axis=0).tolist())
    print ''
    print diri
    print 'test'
    print '\n'.join(tests)
    print ''
    print 'dev'
    print '\n'.join(devs)

# diri = '/home/dpappas/MODELS_OUTPUTS/Model_33_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_34_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_35_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_36_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_37_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_38_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_39_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_40_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_41_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_42_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_43_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_44_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_45_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_46_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_47_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_48_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_49_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_50_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_51_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_52_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_53_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_54_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_55_run_{}/model.log'
# diri = '/home/dpappas/MODELS_OUTPUTS/Model_56_run_{}/model.log'
diri = '/home/dpappas/Model_50_run_5max_{}/model.log'

do_three_losses()
# do_one_loss()





