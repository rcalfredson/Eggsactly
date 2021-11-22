import json
import pickle

with open('/media/Synology3/Robert/'
    'amazon_annots/images/challenging/egg_count_labels_amended_robert.pickle', 'rb') as f:
    pickle_data = pickle.load(f)

with open('validKeys.json') as f:
    valid_keys = json.load(f)

print('keys:', pickle_data.keys())
for k in ('clicks',):
    print('keys in', k)
    input()
    print(pickle_data[k])
    if not hasattr(pickle_data[k], 'keys'):
        continue
    print('quantity:', len(list(pickle_data[k].keys())))
    input()
    for sub_k in list(pickle_data[k].keys()):
        if sub_k not in valid_keys:
            print(sub_k, 'was not found in', k, '?')
            del pickle_data[k][sub_k]

with open('/media/Synology3/Robert/'
    'amazon_annots/images/challenging/egg_count_labels_amended_robert.pickle',
    'wb') as f:
    pickle.dump(pickle_data, f)
