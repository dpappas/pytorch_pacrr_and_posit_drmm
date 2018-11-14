
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

tests, devs = [], []
for i in range(5):
    fpath = diri.format(i)
    if(os.path.exists(fpath)):
        res = {
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
        with open(fpath) as f:
            lines = f.readlines()
            for l in range(len(lines)):
                if('test known MAP documents' in lines[l]):
                    data = lines[l:l+8]
                    # # pprint(data)
                    # res['test']['map_doc']          = float(lines[l+8].split(':')[-1].strip())
                    # res['dev']['map_doc']           = float(lines[l+8].split('epoch_dev_map:')[1].split()[0].strip())
                    # res['epoch']                    = int(lines[l+8].split('epoch:')[1].split()[0])
                    # data                            = [float(t.strip().split()[-1]) for t in data]
                    #
                    res['test']['map_doc_bioasq']   = data[0]
                    res['test']['known_f1_snip']    = data[1]
                    res['test']['known_map_snip']   = data[2]
                    res['test']['known_gmap_snip']  = data[3]
                    res['test']['f1_snip']          = data[5]
                    res['test']['map_snip']         = data[6]
                    res['test']['gmap_snip']        = data[7]
                elif('dev known MAP documents' in lines[l]):
                    data = lines[l:l+8]
                    data = [float(t.strip().split()[-1]) for t in data]
                    #
                    res['dev']['map_doc_bioasq']    = data[0]
                    res['dev']['known_f1_snip']     = data[1]
                    res['dev']['known_map_snip']    = data[2]
                    res['dev']['known_gmap_snip']   = data[3]
                    res['dev']['f1_snip']           = data[5]
                    res['dev']['map_snip']          = data[6]
                    res['dev']['gmap_snip']         = data[7]
            # print(fpath)
            # pprint(res)
            print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                res['test']['map_doc'],
                res['test']['map_doc_bioasq'],
                res['test']['f1_snip'],
                res['test']['map_snip'],
                res['test']['gmap_snip'],
                res['test']['known_f1_snip'],
                res['test']['known_map_snip'],
                res['test']['known_gmap_snip'],
                res['dev']['map_doc'],
                res['dev']['map_doc_bioasq'],
                res['dev']['f1_snip'],
                res['dev']['map_snip'],
                res['dev']['gmap_snip'],
                res['dev']['known_f1_snip'],
                res['dev']['known_map_snip'],
                res['dev']['known_gmap_snip'],
                res['epoch'],
            )
            tests.append(
                '{}\t{}\t{}\t{}\t{}\t{}'.format(
                    res['test']['f1_snip'], res['test']['map_snip'], res['test']['gmap_snip'],
                    res['test']['known_f1_snip'], res['test']['known_map_snip'],
                    res['test']['known_gmap_snip']
                )
            )
            devs.append(
                '{}\t{}\t{}\t{}\t{}\t{}'.format(
                    res['dev']['f1_snip'], res['dev']['map_snip'], res['dev']['gmap_snip'],
                    res['dev']['known_f1_snip'], res['dev']['known_map_snip'], res['dev']['known_gmap_snip']
                )
            )



print ''
print diri
print 'test'
print '\n'.join(tests)
print ''
print 'dev'
print '\n'.join(devs)







