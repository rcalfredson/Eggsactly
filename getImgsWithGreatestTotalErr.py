from glob import glob
import json
import os

error_results = []
json_files = glob(r'P:\Robert\objects_counting_dmap\error_examples\errors*.json')
for f in json_files:
    with open(f) as myF:
        error_results.append(json.load(myF))
with open(r'P:\Robert\objects_counting_dmap\error_examples\error_key_mapping.json') as myF:
    filenameMapping = json.load(myF)

base64_strings = error_results[0].keys()
print('base64 strings?', base64_strings)
error_totals = {b64s:0 for b64s in base64_strings}
print(len(error_totals))
for result_set in error_results:
    for b64s in base64_strings:
        error_totals[b64s] += result_set[b64s]


error_totals = dict(sorted(error_totals.items(), key=lambda item: item[1], reverse=True))
new_error_totals = {}
for key in error_totals:
    new_error_totals[os.path.basename(filenameMapping[key])] = error_totals[key]

with open(r'P:\Robert\objects_counting_dmap\error_examples\error_totals.json', 'w') as myF:
    json.dump(new_error_totals, myF, ensure_ascii=False, indent=4)