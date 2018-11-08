### Input Data Format
1. Protein interaction file (protein.actions.tsv)

    Tab-delimited text file with three columns: proteinA, proteinB, interaction

    proteinA and proteinB - Protein ID. Each of them should contain a corresponding sequence in protein.dictionary.tsv
    interaction - 1 or 0 for binary prediction; class label for type classification; continuous number for affinity prediction

2. Protein sequence file (protein.dictionary.tsv)

    Protein ID and its corresponding sequence are seprated by a tab.

