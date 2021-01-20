import argparse
from collections import Counter
import csv
import pickle

import cv2
import numpy as np

from circleFinder import CircleFinder

def options():
  """Parse options for egg-label deviation calculator."""
  p = argparse.ArgumentParser(description='Calculate standard deviation' +\
      'of human labels for egg-laying regions')
  p.add_argument('files', help='CSV files of egg labels to compare', nargs="*")
  # if we ask for both pickle files and csv files, how to make sure
  # they get paired correctly?
  # one other option: just read the pickle files, since they contain the same
  # information anyway.
  return p.parse_args()

opts = options()
rawData = []
print('files:', opts.files)
for i, fileName in enumerate(opts.files):
    rawData.append([])
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            rawData[i].append(line)

print('rawData:', rawData)
# how best to organize the counts?
# on the highest level, organize by filename (keys)
# assume that the images have been ordered sequentially
# keep a separate dictionary for each CSV, or combine them?
counts = []
for dataList in rawData:
    counts.append(dict())
    countKey = ''
    for row in dataList:
        firstEl = row[0].lower()
        if firstEl.endswith('.jpg') or firstEl.endswith('.png'):
            countKey = firstEl
            counts[-1][countKey] = []
        else:
            counts[-1][countKey].append(row)
for user in counts:
    for fileKey in user:
        user[fileKey] = [count for sublist in user[fileKey] for count in sublist]

goodFiles = Counter([item for d in counts for item in d.keys()])
goodFiles = [fileName for fileName in goodFiles if goodFiles[fileName] > 1]
print('goodFiles:', goodFiles)
# iterate through good files
stdevs = {f: [] for f in goodFiles}
for f in goodFiles:
    # the counts are flattened in each
    # assume that all counts run continuously.
    # calculate standard deviation until hitting first index
    # where there are two empty strings
    # maximum possible index: length of list for any user
    # that opened it
    # open image and segment

    # this script's prior function:
    # collecting half-ranges for each user that labeled it
    # from there, it should flag any image with a half-range greater
    # than a certain absolute value, e.g., 3 eggs.
    maxIndex = -1
    for user in counts:
        if f not in user: continue
        if maxIndex == -1: maxIndex = len(user[f])
        try:
            lastIndex = user[f].index('')
        except ValueError as _:
            lastIndex = len(user[f])
        lastIndex -= 1
        if lastIndex < maxIndex:
            maxIndex = lastIndex
    stdevs[f] = []
    for i in range(maxIndex):
        print('file:', f)
        print('max index:', maxIndex)
        print('trying to calculate std of these vals:')
        samples = []
        for user in counts:
            if f not in user: continue
            print('counts for user:', user[f])
            samples.append(int(user[f][i]))
        stdevs[f].append(0.5*(max(samples) - min(samples)))
        if stdevs[f][-1] > 3:
            print('found an overage in file %s, position %i'%(f, i))
    print('stdevs::', stdevs[f])
    stdevs[f] = stdevs[f]

print('all stdevs:', stdevs)
numCols = 6
with open('varianceReport.txt', 'w') as f:
    for imageName in stdevs:
        f.write('%s\n'%imageName)
        numRows = np.ceil(len(stdevs[imageName]) / numCols).astype(int)
        f.write('mean of std devs: %.2f\n'%np.mean(stdevs[imageName]))
        f.write('individual std devs:\n')
        f.write("\n".join("\t".join(map(str, ['%.2f'%el for el in stdevs[
            imageName][i*numCols:(i + 1)*numCols]])) for i in range(numRows)))
        f.write('\n\n')