
import gc
gc.collect()

import os
from pprint import pprint
# diri = '/home/dpappas/Model_54_run_{}/model.log'
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

tests, devs = [], []
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
                    data = lines[l-16:l+8]
                    # pprint(data)
                    res3['test']['map_doc']         = float(lines[l+8].split(':')[-1].strip())
                    res3['dev']['map_doc']          = float(lines[l+8].split('epoch_dev_map:')[1].split()[0].strip())
                    res3['epoch']                   = int(lines[l+8].split('epoch:')[1].split()[0])
                    data                            = [float(t.strip().split()[-1]) for t in data]
                    #
                    res1['test']['map_doc_bioasq']  = data[0]
                    res1['test']['known_f1_snip']   = data[1]
                    res1['test']['known_map_snip']  = data[2]
                    res1['test']['known_gmap_snip'] = data[3]
                    res1['test']['f1_snip']         = data[5]
                    res1['test']['map_snip']        = data[6]
                    res1['test']['gmap_snip']       = data[7]
                    #
                    res2['test']['map_doc_bioasq']  = data[8]
                    res2['test']['known_f1_snip']   = data[9]
                    res2['test']['known_map_snip']  = data[10]
                    res2['test']['known_gmap_snip'] = data[11]
                    res2['test']['f1_snip']         = data[12]
                    res2['test']['map_snip']        = data[13]
                    res2['test']['gmap_snip']       = data[14]
                    #
                    res3['test']['map_doc_bioasq']  = data[15]
                    res3['test']['known_f1_snip']   = data[16]
                    res3['test']['known_map_snip']  = data[17]
                    res3['test']['known_gmap_snip'] = data[18]
                    res3['test']['f1_snip']         = data[19]
                    res3['test']['map_snip']        = data[20]
                    res3['test']['gmap_snip']       = data[21]
                elif('dev known MAP documents' in lines[l]):
                    data = lines[l:l+8]
                    data = [float(t.strip().split()[-1]) for t in data]
                    #
                    res3['dev']['map_doc_bioasq']    = data[0]
                    res3['dev']['known_f1_snip']     = data[1]
                    res3['dev']['known_map_snip']    = data[2]
                    res3['dev']['known_gmap_snip']   = data[3]
                    res3['dev']['f1_snip']           = data[5]
                    res3['dev']['map_snip']          = data[6]
                    res3['dev']['gmap_snip']         = data[7]
            # print(fpath)
            # pprint(res)
            print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
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
            tests.append(
                '{}\t{}\t{}\t{}\t{}\t{}'.format(
                    res3['test']['f1_snip'],        res3['test']['map_snip'],       res3['test']['gmap_snip'],
                    res3['test']['known_f1_snip'],  res3['test']['known_map_snip'], res3['test']['known_gmap_snip']
                )
            )
            devs.append(
                '{}\t{}\t{}\t{}\t{}\t{}'.format(
                    res3['dev']['f1_snip'],         res3['dev']['map_snip'],        res3['dev']['gmap_snip'],
                    res3['dev']['known_f1_snip'],   res3['dev']['known_map_snip'],  res3['dev']['known_gmap_snip']
                )
            )



print ''
print diri
print 'test'
print '\n'.join(tests)
print ''
print 'dev'
print '\n'.join(devs)







