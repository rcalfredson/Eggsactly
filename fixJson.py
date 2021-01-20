import json

from utils.coco import getNewCategory

with open('generated/ellipseDataset.json') as f:
    cocoData = json.load(f)

for cat in range(1, 21):
    cocoData['categories'].append(getNewCategory(cat))

with open('generated/fixedDataset.json', 'w') as f:
    json.dump(cocoData, f, ensure_ascii=False, sort_keys=True, indent=4)