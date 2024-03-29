# Evaluating Embeddings for Omicron Lineage Identification
The provided codes are designed to compare various embeddings from the literature, which represent the spike protein, to identify Dominant Omicron Lineages (DOL). DOL are defined as those that reach a frequency of at least 10% in the dataset for a given week. This criterion has led to the identification of 16 DOL, including BA.1, BA.2, BA.5, BA.2.12.1, XBB.1.5, and EG.5.1. Initially, BA.1 was prevalent and triggered the fourth wave, but it was subsequently replaced by BA.2, which is significantly more transmissible. Additionally, BA.5 and BA.12.1 show stronger evasion of the host's immune response, while XBB.1.5, identified in 2023, exhibits increased infectivity due to enhanced binding to the ACE-2 receptor for SARS-CoV-2.

**Objective**
The goal of this Python codes are to compare various spike protein embeddings and anomaly detection models to determine the most effective method for representing and identifying DOL. The embeddings include methods for numerical representation, such as Spike2Vec, k-mers, Autoencoder, and Wasserstein Distance Guided Representation Learning, as well as numerical signals like Spike2Sig.

# Python Packages
The Python packages needed are : 
1. pandas (1.5.3)
2. numpy (1.24.1)
3. statistics (1.0.3.5)
4. scikit-learn (1.2.1)
5. scipy (1.10.1)
6. matplotlib (3.8.3)
7. fnvhash (0.1.0)
8. keras (2.14.0)
9. tensorflow (2.14.0)
10. torch (1.10.2)


Python 3.9 is required.

# Dataset Generation
We create a different Dataset for each different embeddig. The codes receive as input the file spikes.fasta and metadataset.csv downloaded from the [GISAID site](https://gisaid.org). In the folder <code>DatasetGeneration</code> are contained the codes to create dataset. In particular:
1. <code>Embedding_Global_dataset.py</code> : Is the main code to generate the different datasets;
2. <code>utils</code> : Contains the functions used in the main code.
The code inputs consist of the file.fasta (<code>Spikes.fasta</code>) that contains the amino acid sequences and the file.csv (<code>metadataset.csv</code>) that contains the characteristics of each sequence. Both files can be downloaded from [GISAID](https://gisaid.org)

# Parameter Tuning 
We performed parameter tuning using a grid search method, taking into account the temporal progression of the data across the four trimesters of 2020. In essence, we trained our models iteratively on an increasing amount of data - first on the first trimester, then on the first and second trimesters, and finally on the first, second, and third trimesters. We then tested the models on the remaining trimesters (second to fourth, third and fourth, and fourth, respectively). The selection of the best parameter combinations for each model was based on the median balanced accuracy.

**Folder Struture**
Each primary folder <code>ParamTuning</code> contains code for a specific type of embedding. Inside these primary folders, there are subfolders, each dedicated to a different anomaly detection model. Within each of these subfolders, there are two key files:

1. <code>Main.py</code>: This file contains the main pipeline for the model.
2. <code>Utils.py</code>: This file includes various functions that are utilized in <code>Main.py</code>.

The inputs for the code are derived from the outputs of previous code <code>Embedding_Global_dataset.py</code> executions
# Training and Evaluation
Our models were trained on the GISAID Spike proteins from 2021, with a particular emphasis on the Alpha and Delta variants. For the testing phase, we utilized sequences from 2022 and 2023 (up until November 8), a period marked by the spread of the Omicron variants. The models’ performance was evaluated using 16 dominant Omicron lineages, which were identified as true positives. Key metrics such as Precision (Pr), False Positive Rate (Fpr), and Balanced Accuracy (Ba) were calculated for this evaluation.

**Folder Structure**
Each primary folder is dedicated to a specific type of embedding. Within these primary folders, there are numerous subfolders. Each of these subfolders contains the code for a different anomaly detection model.Within each of these subfolders, there are two key files:

1. <code>Main.py</code>: This file contains the main pipeline for the model.
2. <code>Utils.py</code>: This file includes various functions that are utilized in <code>Main.py</code>.

The inputs for the code are derived from the outputs of previous code <code>Embedding_Global_dataset.py</code> executions



