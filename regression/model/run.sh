cd rcnn
mkdir results
mkdir records
# python program.py <dataset> <column_index_of_score> <result_location> <id_for_embeddings: 0=onehot, 1=a_c, 2=a_{ph}, 3=[a_c,a_{ph}]> <hidden_dim> <epochs_per_fold>
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../lss/preprocessed/prcoessed_scores.tsv -1 results/lss_wvctc_rcnn_25.txt 3 25 200
CUDA_VISIBLE_DEVICES=4 python rcnn.py ../../../lss/preprocessed/prcoessed_scores.tsv -1 results/lss_wvctc_rcnn_50.txt 3 50 200
