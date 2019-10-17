cd rcnn
mkdir results

#0.25
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.25.tsv -1 results/0.25_yeast_wvctc_rcnn_50_5.txt 3 50 150
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.25.tsv -1 results/0.25_yeast_wvctc_rcnn_25_5.txt 3 25 150
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.25.tsv -1 results/0.25_yeast_wvctc_rcnn_75_5.txt 3 75 150


# 0.1
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.10.tsv -1 results/0.1_yeast_wvctc_rcnn_50_5.txt 3 50 150
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.10.tsv -1 results/0.1_yeast_wvctc_rcnn_25_5.txt 3 25 150
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.10.tsv -1 results/0.1_yeast_wvctc_rcnn_75_5.txt 3 75 150


CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.01.tsv -1 results/0.01_yeast_wvctc_rcnn_50_5.txt 3 50 150
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.01.tsv -1 results/0.01_yeast_wvctc_rcnn_25_5.txt 3 25 150
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.01.tsv -1 results/0.01_yeast_wvctc_rcnn_75_5.txt 3 75 150


#0.4
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.40.tsv -1 results/0.4_wvctc_rcnn_25_5.txt 3 25 150
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.filtered.40.tsv -1 results/0.4_wvctc_rcnn_50_5.txt 3 50 150


#all
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.tsv -1 results/all_wvctc_rcnn_50_5.txt 3 50 150
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.tsv -1 results/all_wvctc_rcnn_25_5.txt 3 25 150


