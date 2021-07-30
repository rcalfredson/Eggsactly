import pickle

with open(r'P:\Egg images_9_3_2020\egg_count_labels_uli.pickle', 'rb') as f:
# with open('C:/Users/Tracking/counting-3/imgs/Charlene/debug/egg_labels_robert.pickle', 'rb') as f:
    tempData = pickle.load(f)

print('pickle data?')
print([k for k in tempData['clicks'] if '10_15_2020_img_0003.jpg' in k])
print('frontier data:', tempData['frontierData']['fileName'])
print('finished labeling?', tempData['frontierData']['finishedLabeling'])
# print(tempData['frontierIndex'])
exit(0)
for clickKey in tempData['clicks'].keys():
    if 'jul10_6right.jpg' in clickKey:
        print(clickKey)
competingKeys = ('jul10_6right.jpg_-1_5',
                 'jul10_6right.jpg_0_5')
for key in competingKeys:
    print('clicks for', key)
    print(tempData['clicks'][key])