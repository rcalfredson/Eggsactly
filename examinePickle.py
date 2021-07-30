import argparse
from glob import glob
import os
import pickle
from clickLabelManager import ClickLabelManager
import platform

invalid_keys = (
    "img_0002.jpg_0_0_right",
    "img_0002.jpg_0_1_upper",
    "img_0002.jpg_0_1_left",
    "img_0002.jpg_1_0_lower",
    "img_0002.jpg_1_1_right",
    "2020-11-20_img_0002.jpg_0_0",
    "2020-11-20_img_0002.jpg_1_0",
    "2020-12-03_img_0001.jpg_0_0",
    "2020-12-03_img_0001.jpg_1_0",
    "2020-11-21_img_0001.jpg_0_0",
    "2020-11-21_img_0001.jpg_0_0",
    "2020-11-21_img_0001.jpg_1_0",
    "2020-12-02_img_0007.jpg_0_0",
    "2020-12-02_img_0007.jpg_1_0",
    "2020-11-24_img_0008.jpg_0_0",
    "2020-11-24_img_0008.jpg_1_0",
)

p = argparse.ArgumentParser(description="fix legacy names in pickle files")
p.add_argument("dir", help="path to folder containing pickle files to update")
args = p.parse_args()

pickle_files = glob(f"{args.dir}/*metadata")
print("pickle files:", pickle_files)
input()

# if platform.system() == 'Windows':
#   base_path = r"P:\Egg images_9_3_2020\WT_5"
# elif platform.system() == 'Linux':
#   base_path = '/media/Synology3/Egg images_9_3_2020/WT_5'

# pickle_file = 'tempHide/egg_count_labels_robert.pickle'
chamberTypes = "chamberTypes"

old_to_new = {
    "new": "opto",
    "old": "sixByFour",
    "threeBy5": "fiveByThree",
    "fourCircle": "large",
}


def clean_up_chamber_type_names(f_name):
    with open(os.path.join(f_name), "rb") as f:
        loaded = pickle.load(f)
        print("keys:", loaded.keys())
        for k in loaded.keys():
            print(f"Information for {k}:")
            print(loaded[k])
            if loaded[k]["ct"] in old_to_new:
                print(
                    (
                        f"Changing {loaded[k]['ct']} "
                        f"to {old_to_new[loaded[k]['ct']]}"
                    )
                )
                loaded[k]['ct'] = old_to_new[loaded[k]['ct']]
                print('new value:', loaded[k]['ct'])

    #     for k in loaded[chamberTypes]:
    #         if loaded[chamberTypes][k] in old_to_new:
    #             print((
    #                 f"Changing {loaded[chamberTypes][k]} "
    #                 f"to {old_to_new[loaded[chamberTypes][k]]}")
    #             )
    #             loaded[chamberTypes][k] = old_to_new[loaded[chamberTypes][k]]
    #             print('new value:', loaded[chamberTypes][k])

    with open(f_name, "wb") as f:
        pickle.dump(loaded, f)
        print(f"Wrote new copy of file {f_name}.")


for f_name in pickle_files:
    clean_up_chamber_type_names(f_name)


# for keyType in loaded.keys():
#         print('items for key type', keyType)
#         if not hasattr(loaded[keyType], 'keys'):
#           continue
#         print(loaded[keyType].keys())
#         print(len(loaded[keyType].keys()))
#         for k in list(loaded[keyType].keys()):
#           print(f"Key {k} of {keyType}:")
#           print(loaded[keyType][k])
#           if type(loaded[keyType][k]) is list:
#             print(len(loaded[keyType][k]))
#           if k in invalid_keys:
#             del loaded[keyType][k]
#             print('deleted bad key', k)
#             input()
#         input()
#       input()
#       print('what is loaded clicks?', loaded['clicks'].keys())

#       # for key in list(loaded['clicks'].keys()):
#       #   for basename in optogeneticBasenames:
#       #     if basename in key:
#       #       del loaded['clicks'][key]
#       print('and its type:', type(loaded))
