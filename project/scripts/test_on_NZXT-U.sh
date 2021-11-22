#!/bin/bash
cd /media/Synology3/Robert/counting-3
PATH=/usr/local/bin:$PATH
source /home/robert/anaconda3/bin/activate torch-py37
python scripts/test_counting_page.py http://10.122.168.56:5000\
 static/test_data/egg_ct_official_2021_10_30_img_0002.csv\
  /media/Synology3/Rebecca/2021_10_30/IMG_0002.JPG rca30@duke.edu