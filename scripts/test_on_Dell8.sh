#!/bin/bash
cd /media/Synology3/Robert/counting-3
PATH=/usr/local/bin:$PATH
source /home/robert/miniconda3/bin/activate torchEnv
python scripts/test_counting_page.py http://10.122.168.56:5000\
 watch parallel rca30@duke.edu