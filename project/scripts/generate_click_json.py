import argparse
from glob import glob
import json
import os
import pickle

p = argparse.ArgumentParser(description='')
p.add_argument('pickled_click_data',
    help='pickle file containing the click data to be converted to JSON.')
p.add_argument('dest',
help='folder where the JSON files will be written (must already exist)')

opts = p.parse_args()

with open(opts.pickled_click_data, 'rb') as f:
    pickled_data = pickle.load(f)

print('pickled data?', pickled_data.keys())
print('chamber types', pickled_data['chamberTypes'])
with open('debug1', 'w') as f:
    print('all click data:', pickled_data['clicks'], file=f)
input()
print('file names with clicks?', pickled_data['clicks'].keys())
print('how many file names?', len(list(pickled_data['clicks'].keys())))
sub_images_list = [os.path.basename(el.split('.jpg')[0]) for el in glob(
    '/media/Synology3/Robert/amazon_annots/'
    'images/challenging/subimgs/*jpg')]
pickled_filenames = [el.replace('.jpg', '') for el in list(pickled_data['clicks'].keys())]
print('subimages list?', len(sub_images_list))
image_name_diff = list(set(pickled_filenames) - set(sub_images_list))
print('keys in one but not the ohter:', image_name_diff)
print(len(image_name_diff))
input()
for k in pickled_data['clicks'].keys():
    clicks_filename = f"{k.replace('.jpg', '')}_clicks.json"
    print('clicks for', clicks_filename)
    with open(os.path.join(opts.dest, clicks_filename), 'w') as f:
        json.dump(pickled_data['clicks'][k], f)
        # remove the ".jpg" from the filename
        print(pickled_data['clicks'][k])
