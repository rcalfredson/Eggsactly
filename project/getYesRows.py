import argparse

def options():
  """Parse options for sub-image detector."""
  p = argparse.ArgumentParser(description=
    'Find chambers within an egg-laying image and write each one as a ' +
    'separate image.')
  p.add_argument('table', help='tab-delimited table containing yes and no responses')
  return p.parse_args()

opts = options()
with open(opts.table) as f:
  data = f.read().splitlines()

keptNames = []

for datum in data:
  splitData = datum.split('\t')
  print('split data:', splitData)
  if 'yes' in splitData[-1]:
    keptNames.append(splitData[0])

print('kept names:', keptNames)
with open('keptNames.txt', 'w') as f:
  [f.write('%s\n'%keptName) for keptName in keptNames]