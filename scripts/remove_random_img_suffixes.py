from glob import glob
import os

images = glob("/media/Synology3/Egg images_9_3_2020/with_larvae/subimgs/*.jpg")

for img_name in images:
    new_filename = f"{'_'.join(img_name.split('_')[:-1])}.jpg"
    print(
        "Will rename", os.path.basename(img_name), "to", os.path.basename(new_filename)
    )

    os.rename(img_name, new_filename)
    print("Created filename", new_filename)
