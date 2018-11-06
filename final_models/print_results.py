
import os
from pprint import pprint
diri = '/home/dpappas/Model_54_run_{}/model.log'

for i in range(5):
    fpath = diri.format(i)
    if(os.path.exists(fpath)):
        res = {
            'dev': {
                'map_doc'           : 0.,
                'f1_snip'           : 0.,
                'map_snip'          : 0.,
                'gmap_snip'         : 0.,
                'known_f1_snip'     : 0.,
                'known_map_snip'    : 0.,
                'known_gmap_snip'   : 0.,
            },
            'test': {
                'map_doc'           : 0.,
                'f1_snip'           : 0.,
                'map_snip'          : 0.,
                'gmap_snip'         : 0.,
                'known_f1_snip'     : 0.,
                'known_map_snip'    : 0.,
                'known_gmap_snip'   : 0.,
            },
            'epoch'                 : 0.,
            'time'                  : 0.,
        }
        dev_known_map_document  = 0.
        dev_known_f1_snips      = 0.
        dev_known_map_snips     = 0.
        dev_known_gmap_snips    = 0.
        dev_map_document        = 0.
        dev_f1_snips            = 0.
        dev_map_snips           = 0.
        dev_gmap_snips          = 0.
        with open(fpath) as f:
            lines = f.readlines()
            for l in range(len(lines)):
                if('test known MAP documents' in lines[l]):
                    data = lines[l:l+9]
                    data = [float(t.strip().split()[-1]) for t in data]
                    pprint(data)
                elif('dev known MAP documents' in lines[l]):
                    data = lines[l:l+9]
                    data = [float(t.strip().split()[-1]) for t in data]
                    pprint(data)
























