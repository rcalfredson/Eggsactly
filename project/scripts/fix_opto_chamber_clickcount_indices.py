from glob import glob
import json
import pickle

with open(
    "/media/Synology3/Robert/"
    "amazon_annots/images/challenging/egg_count_labels_robert.pickle",
    "rb",
) as f:
    pickle_data = pickle.load(f)

with open("validKeys.json") as f:
    valid_keys = json.load(f)

print("keys:", pickle_data.keys())
print("chamber types:", pickle_data["chamberTypes"])
opto_chamber_types = [
    el.split('.jpg')[0]
    for el in pickle_data["chamberTypes"].keys()
    if pickle_data["chamberTypes"][el] == "opto"
]
print("opto chamber types:", opto_chamber_types)
print('how many click keys before?', len(list(pickle_data['clicks'].keys())))
input()
new_clicks = {}
for k in ("clicks",):
    for sub_k in list(pickle_data[k].keys()):
        is_opto = False
        match_string = sub_k.split('.jpg')[0]
        for img in opto_chamber_types:
            if img == match_string:
                is_opto = True
                break
        if is_opto:
            split_key = sub_k.split('_')
            row_orig, col_orig = map(int, split_key[-2:])
            start_of_key = '_'.join(split_key[:-2])
            new_row = row_orig * 2 + col_orig // 5
            new_col = col_orig % 5
            new_key = f"{start_of_key}_{new_row}_{new_col}"
            # print('new key:', new_key)
            # input()
            new_clicks[new_key] = list(pickle_data[k][sub_k])
        else:
            new_clicks[sub_k] = list(pickle_data[k][sub_k])
pickle_data['clicks'] = new_clicks
print('after?', len(list(pickle_data['clicks'].keys())))
print(len(new_clicks.keys()))
with open(
    "/media/Synology3/Robert/"
    "amazon_annots/images/challenging/egg_count_labels_amended_robert.pickle",
    "wb",
) as f:
    pickle.dump(pickle_data, f)
