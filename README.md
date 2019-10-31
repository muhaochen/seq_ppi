## Multifaceted Protein-Protein Interaction Prediction Based on Siamese Residual RCNN

This is the repository for PIPR (originally Lasagna in our previous preprint version). This repository contains the source code and links to some datasets used in our paper.

Environment:

    python 2.7 or 3.6
    Tensorflow 1.7 (with GPU support)
    CuDNN
    Keras 2.2.4
    
./binary contains the implementation for the binary prediction task.  
./type contains that for the interaction type prediction task.  
./regression contains that for the binding affinity prediction task. 
Each folder is attached with a run.sh to show how to run the evaluation program.  
./embeddings contains pre-trained amino acid embeddings and the training script.  

## Datasets

Here we include altogether 6 datasets. New datasets processed in this work are marked with *ND*
1. The Yeast dataset for binary PPI prediction.  
2. The *multi-species dataset (C. elegans, D. melanogaster and E. coli)* for binary PPI prediction. (*ND*)  
3. Added another binary PPI prediction dataset from \[Pan et el. 2010\] under the folder *sun*.  
4-5. The SHS27k and SHS148k datasets for interaction type prediction can be downloaded from [here](http://yellowstone.cs.ucla.edu/~muhao/pipr/SHS_ppi_beta.zip) or from the [Google Drive](https://drive.google.com/open?id=1y_5gje6AofqjrkMPY58XUdKgDuu1mZCh). (*ND*)  
6. Link to the normalized SKEMPI dataset is [here](http://yellowstone.cs.ucla.edu/~muhao/pipr/SKEMPI_all_dg_avg_(PIPR).zip).  

## Reference
This work has been published in the Bioinformatics journal featuring ISMB/ECCB 2019.

DOI: http://dx.doi.org/10.1093/bioinformatics/btz328  
Bibtex:

    @article{chen2019pipr,
        title={Multifaceted Protein-Protein Interaction Prediction Based on Siamese Residual RCNN},
        author={Chen, Muhao and Ju, Chelsea and Zhou, Guangyu and Chen, Xuelu and Zhang, Tianran and Chang, Kai-Wei and Zaniolo, Carlo and Wang, Wei},
        journal={Bioinformatics},
        volume = {35},
        number = {14},
        pages = {i305-i314},
        year = {2019},
        month = {07},
        publisher={Oxford University Press}
    }
