import pickle

with open('C:/Users/Tracking/counting-3/imgs/Charlene/temp2/egg_labels_robert.pickle', 'rb') as f:
# with open('C:/Users/Tracking/counting-3/imgs/Charlene/debug/egg_labels_robert.pickle', 'rb') as f:
    tempData = pickle.load(f)

print('pickle data?')
print(tempData['frontierIndex'])
exit(0)
for clickKey in tempData['clicks'].keys():
    if 'jul10_6right.jpg' in clickKey:
        print(clickKey)
competingKeys = ('jul10_6right.jpg_-1_5',
                 'jul10_6right.jpg_0_5')
for key in competingKeys:
    print('clicks for', key)
    print(tempData['clicks'][key])