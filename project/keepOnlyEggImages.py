import json

with open('C:\\Users\\Tracking\\Downloads\\response_1603909087679.json') as f:
    imageData = json.load(f)

print('imageData before cleanup', imageData['images'][0:15])
for i, image in enumerate(list(imageData['images'])):
    if image['dataset_id'] != 10:
        imageData['images'].remove(image)

print('and after:', imageData['images'][0:15])
with open('cocoImageMap.json', 'w') as f:
    json.dump(imageData['images'], f, ensure_ascii=False, indent=4)
