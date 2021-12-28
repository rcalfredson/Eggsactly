sed -i.bak '/torch/d' ./requirements.txt
gcloud app deploy --project=egg-counting-336514 --quiet
mv -f requirements.txt.bak requirements.txt
