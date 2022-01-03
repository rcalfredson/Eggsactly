sed -i.bak '/torch/d' ./requirements.txt
sed -i '/opencv-python/c\opencv-python-headless' ./requirements.txt
mv .env .env.bak
mv .env_aws_deploy .env
docker build --tag egg-counting .
mv -f requirements.txt.bak requirements.txt
mv .env .env_aws_deploy
mv .env.bak .env
