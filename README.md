## Multifaceted Protein-Protein Interaction Prediction Based on Siamese Residual RCNN

This is the repository for PIPR (originally Lasagna in our previous preprint version). This repository is contains the source code and links to some datasets used in our paper. (to be updated)

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

The link to the STS datasets is [here](http://yellowstone.cs.ucla.edu/~muhao/pipr/SHS_ppi_beta.zip).

## Reference
This paper is going to appear in the Bioinformatics journal featuring ISMB/ECCB 2019. (* indicates co-first authors)
Muhao Chen*, Chelsea J. T. Ju*, G. Zhou, X. Chen, T. Zhang, Kai-Wei Chang, Carlo Zaniolo, and Wei Wang. Multifaceted Protein-Protein Interaction Prediction Based on Siamese Residual RCNN. To appear in Bioinformatics. 

Temporary bibtex:

    @article{chen2019pipr,
        title={Using genomic annotations increases statistical power to detect eGenes},
        author={Chen, Muhao and Ju, Chelsea and Zhou, Guangyu and Chen, Xuelu and Zhang, Tianran and Chang, Kai-Wei and Zaniolo, Carlo and Wang, Wei},
        journal={Bioinformatics (to appear)},
        year={2019},
        publisher={Oxford University Press}
    }
