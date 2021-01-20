import json, os

# PRIOR_OFFSET = 4618
# OFFSET = 2458
base = 'C:\\Users\\Tracking\\counting-3\\imgs\\Charlene\\temp2\\results'
fileName = 'transformedEggLabels.json'

with open(os.path.join(base, fileName)) as f:
    data = json.load(f)

with open('cocoImageMap.json') as f:
    imageMap = json.load(f)
newImageMap = dict()
for image in imageMap:
    newImageMap[image['file_name']] = image['id']
imageMap = newImageMap

# create a map of fileNames to the image IDs in labels file
fileNameToIdMap = dict()
for image in data['images']:
    fileNameToIdMap[image['id']] = image['file_name']

for i in range(len(data['annotations'])):
    data['annotations'][i]['image_id'] = imageMap[fileNameToIdMap[data['annotations'][i]['image_id']]]
for i in range(len(data['images'])):
    data['images'][i]['id'] = imageMap[fileNameToIdMap[data['images'][i]['id']]]
    # data['images'][i]['path'] = '/datasets/eggs/' + os.path.basename(data['images'][i]['path'])
    # data['images'][i]['file_name'] += '.jpg'
    # data['images'][i]['file_name'] = os.path.basename(data['images'][i]['path'])

with open(os.path.join(base, "fixed_%s"%fileName), 'w') as f:
  json.dump(data, f, ensure_ascii=False, indent=4)