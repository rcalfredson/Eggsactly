sed -i.bak '/torch/d' ./requirements.txt
sed -i '/opencv-python/c\opencv-python-headless' ./requirements.txt
sudo docker build --tag egg-counting .
mv -f requirements.txt.bak requirements.txt
