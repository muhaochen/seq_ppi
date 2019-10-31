cd lasagna
mkdir results
# python program.py <dataset> <column_index_of_label> <result_location> <id_for_embeddings: 0=onehot, 1=a_c, 2=a_{ph}, 3=[a_c,a_{ph}]> <hidden_dim> <epochs_per_fold>
CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../yeast/preprocessed/protein.actions.tsv -1 results/yeast_wvctc_rcnn_50_5.txt 3 50 100
CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../yeast/preprocessed/protein.actions.tsv -1 results/yeast_wvctc_rcnn_25_5.txt 3 25 100
CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../yeast/preprocessed/protein.actions.tsv -1 results/yeast_wvctc_rcnn_75_5.txt 3 75 100
