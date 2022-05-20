# Set dst repo here.
repo="T3-1_code"
mkdir ../${repo}
mkdir ../${repo}/evaluation
mkdir ../${repo}/models

cp ./*.sh ../${repo}
cp ./*.py ../${repo}
cp ./evaluation/*.py ../${repo}/evaluation
cp ./models/*.py ../${repo}/models
# mv ../${repo}/gco.sh ../${repo}/${repo}_gco.sh
