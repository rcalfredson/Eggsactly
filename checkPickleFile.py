import pickle

with open(r"P:\Egg images_9_3_2020\WT_5\egg_count_labels_robert.pickle", 'rb') as my_f:
    labels = pickle.load(my_f)

print('data?', labels['isBlurryLabels'].keys())
for key in ['img_0002.jpg_0_0_right', 'img_0002.jpg_0_1_upper', 'img_0002.jpg_0_1_left', 'img_0002.jpg_1_0_lower', 'img_0002.jpg_1_1_right']:
    labels['clicks']['2020-11-20_%s'%key] = labels['clicks'][key]
    del labels['clicks'][key]
with open (r"P:\Egg images_9_3_2020\WT_5\egg_count_labels_robert_manual_amend.pickle", 'wb') as my_f:
    pickle.dump(labels, my_f)
# print('keys that start with img?', [k for k in labels['clicks'].keys() if k.startswith('img')])
# print('keys that start with 2020-11-24_img_0008',
#     [k for k in data['clicks'].keys() if k.startswith('2020-11-24_img_0008')])
# with open(r"P:\Egg images_9_3_2020\WT_5\images.metadata", 'rb') as my_f:
#     data = pickle.load(my_f)

# print('image metadata keys?', data.keys())
# print('egg labels keys?', labels['clicks'].keys())