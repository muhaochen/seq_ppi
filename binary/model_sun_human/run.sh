# This runs the 10-fold CV on Pan's human PPI dataset. This is not a part of the results in our ISMB paper, but here are what we've got:
# Acc=98.11 Prec=98.99 F1=98.03

cd ../lasagna
mkdir results

CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../sun/preprocessed/Supp-AB.tsv -1 results/sun_wvctc_rcnn_25_5.txt 3 25 100
CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../sun/preprocessed/Supp-AB.tsv -1 results/sun_wvctc_rcnn_50_5.txt 3 50 100
CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../sun/preprocessed/Supp-AB.tsv -1 results/sun_wvctc_rcnn_75_5.txt 3 75 100
