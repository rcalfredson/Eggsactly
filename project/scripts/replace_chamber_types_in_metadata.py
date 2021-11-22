import json
import pickle

with open('/media/Synology3/Robert/'
    'amazon_annots/images/challenging/images.metadata', 'rb') as f:
    pickle_data = pickle.load(f)

print('keys:', pickle_data.keys())
for k in pickle_data.keys():
    print('key of a single image:', pickle_data[k])
