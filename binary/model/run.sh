cd rcnn
mkdir results
CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../yeast/preprocessed/protein.actions.tsv -1 results/yeast_wvctc_rcnn_50_5.txt 3 50 100
CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../yeast/preprocessed/protein.actions.tsv -1 results/yeast_wvctc_rcnn_50_5.txt 3 25 100
