
import json
from pprint import pprint

d1 = json.load(open('C:\\Users\\dvpap\\OneDrive\\Desktop\\batch3_submit_files\\ir_results\\system1_output_b3\\v3 test_emit_bioasq.json'))
d2 = json.load(open('C:\\Users\\dvpap\\OneDrive\\Desktop\\batch3_submit_files\\ir_results\\system2_output_b3\\v3 test_emit_bioasq.json'))

for q in d1['questions']:
    pprint(q)
    exit()



