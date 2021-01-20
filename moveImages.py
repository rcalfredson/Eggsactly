import glob, os, math, shutil

origin = r'P:\Charlene\egg-laying pic\WT'
dest = r'P:\Egg images_9_3_2020'
with open('keptNames.txt') as f:
    fileNames = f.read().splitlines()

imgFound = 0
for i, fileName in enumerate(fileNames):
    print('checking this path', "%s*"%os.path.join(origin, fileName))
    globResult = glob.glob("%s*"%os.path.join(origin, fileName))
    if len(globResult) == 0: continue
    imgFound += 1
    image = globResult[0]
    print('image is', image)
    destFolder = 'WT_%i'%(math.floor(i / 10))
    print('destFolder:', destFolder)
    shutil.copy(image, os.path.join(dest, destFolder, os.path.basename(image)))

print('images found:', imgFound)
    # Sept2_12right
    # Sept2_s6left
