#!/bin/bash
cd /media/Synology3/Robert/counting-3-debug
PATH=/usr/local/bin:$PATH
source /home/robert/miniconda3/bin/activate torchEnv
python project/scripts/test_counting_page.py http://10.122.168.56:5000\
 watch parallel rca30@duke.edu /media/Synology3/Robert/counting-3