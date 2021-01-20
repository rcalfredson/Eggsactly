import argparse, csv, os

from common import globFiles
from pascalVocHelper import PascalVOCHelper

def options():
  """Parse options for CSV object-count exporter."""
  p = argparse.ArgumentParser(description=
    'Export a CSV file containing image filenames and counts of objects of a single class.')
  p.add_argument('-d', dest='dir', help='directory containing .png images')
  return p.parse_args()

def create_object_count_labels_csv(imgDir):
    imgPaths = globFiles(imgDir)
    objectLabel = PascalVOCHelper('%s.xml'%imgPaths[0].split('.')[0]).objLbl
    with open('counting_%s.csv'%objectLabel, 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', objectLabel])
        for path in imgPaths:
            writer.writerow([os.path.basename(path).split('.png')[0],
                len(PascalVOCHelper('%s.xml'%path.split('.')[0]).boundingBoxes())])

opts = options()
create_object_count_labels_csv(opts.dir)

    # path_labels = os.path.join(root, 'devkit', dataset, 'csv')
    # i = 0

    # img_list_txt = os.path.join(root, 'devkit', dataset, 'ImageSets', 'Main', '%s.txt' % (set))
    # print(img_list_txt)
    # with open(img_list_txt, 'r') as rf:
    #     lines = rf.readlines()
    # text_file = open(file_csv, "w")
    # text_file.write("name,eggs\n")
    # for line in lines:  # assuming gif
    #     i += 1
    #     filename = path_labels + '/' + line[:-1] + '.csv'
    #     with open(filename) as f:
    #         data = f.readlines()
    #         data = data[1:]
    #         img_name = filename[len(path_labels) + 1:-4]
    #         count_label = int(len(data))
    #         text_file.write(("%s,%d\n" % (img_name, count_label)))

    # text_file.close()