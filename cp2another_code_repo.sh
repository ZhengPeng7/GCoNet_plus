repo="bsl_aug_BI_mask_db300"
mkdir ../${repo}
mkdir ../${repo}/evaluation
mkdir ../${repo}/models

cp ../GCoNet_ext/*.sh ../${repo}
cp ../GCoNet_ext/*.py ../${repo}
cp ../GCoNet_ext/evaluation/*.py ../${repo}/evaluation
cp ../GCoNet_ext/models/*.py ../${repo}/models
mv ../${repo}/gco.sh ../${repo}/${repo}_gco.sh
