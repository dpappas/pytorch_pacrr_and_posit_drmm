
import gc
gc.collect()

import os
from pprint import pprint
fpath = '/home/dpappas/this_is_me_testing_Model_41/model.log'
# fpath = '/home/dpappas/this_is_me_testing_Model_42/model.log'
# fpath = '/home/dpappas/this_is_me_testing_Model_43/model.log'
# fpath = '/home/dpappas/this_is_me_testing_Model_44/model.log'

# fpath = '/home/dpappas/this_is_me_testing_Model_49/model.log'
# fpath = '/home/dpappas/this_is_me_testing_Model_50/model.log'
# fpath = '/home/dpappas/this_is_me_testing_Model_51/model.log'
# fpath = '/home/dpappas/this_is_me_testing_Model_52/model.log'
# fpath = '/home/dpappas/this_is_me_testing_Model_52/model.log'

tests, devs = [], []
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
                # pprint(data)
                data                            = [float(t.strip().split()[-1]) for t in data]
                #
                res['test']['map_doc_bioasq']   = data[0]
                res['test']['known_f1_snip']    = data[1]
                res['test']['known_map_snip']   = data[2]
                res['test']['known_gmap_snip']  = data[3]
                res['test']['f1_snip']          = data[5]
                res['test']['map_snip']         = data[6]
                res['test']['gmap_snip']        = data[7]
                tests.append(
                    '{}\t{}\t{}\t{}\t{}\t{}'.format(
                        res['test']['f1_snip'],         res['test']['map_snip'],        res['test']['gmap_snip'],
                        res['test']['known_f1_snip'],   res['test']['known_map_snip'],  res['test']['known_gmap_snip']
                    )
                )
                devs.append(
                    '{}\t{}\t{}\t{}\t{}\t{}'.format(
                        res['dev']['f1_snip'], res['dev']['map_snip'], res['dev']['gmap_snip'],
                        res['dev']['known_f1_snip'], res['dev']['known_map_snip'], res['dev']['known_gmap_snip']
                    )
                )
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

print fpath
print 'test'
print '\n'.join(tests)
print ''
print 'dev'
print '\n'.join(devs)





