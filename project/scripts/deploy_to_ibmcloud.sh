sed -i.bak '/torch/d' ./requirements.txt
sed -i '/opencv-python/c\opencv-python-headless' ./requirements.txt
ibmcloud cf push
mv -f requirements.txt.bak requirements.txt