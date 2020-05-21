# Predicting whether a DNA sequence region is binding site to a specific transcription factor

This is a data challenge for the course "Kernel Methods" for AMMI 2020

## Description
### Introduction
The goal of the data challenge is to learn how to implement machine learning algorithms, gain understanding about them and adapt them to structural data.
For this reason, we have chosen a sequence classification task: predicting whether a DNA sequence region is binding site to a specific transcription factor.

Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.
In this challenge, we will work with three datasets corresponding to three different TFs.

### What is expected
Two days after the deadline of the data challenge, you will have to provide
    - a small report on what you did (in pdf format, 11pt, 2 pages A4 max)
    - your source code (zip archive), with a simple script "start" (that may be called from Matlab, Python, R, or Julia) which will reproduce your submission and saves it in Yte.csv
### Rules
    - At most 2 persons per team. To create a team, you can go to section "Team" after registration.
    - One team can submit results up to fifth per day during the challenge.
    - A leader board will be available during the challenge, which shows the best results per team, as measured on a subset of the test set. A different part of the test set will be used after the challenge to evaluate the results.
    - The most important rule is: DO IT YOURSELF. The goal of the data challenge is not get the best recognition rate on this data set at all costs, but instead to learn how to implement things in practice, and gain practical experience with the machine learning techniques involved.

For this reason, the **use of external machine learning libraries is forbidden**. For instance, this includes, but is not limited to, libsvm, liblinear, scikit-learn, …

On the other hand, you are welcome to use general purpose libraries, such as library for linear algebra (e.g., svd, eigenvalue decompositions), optimization libraries (e.g., for solving linear or quadratic programs)



## Data Description

### Principal Files
This data challenge contains one dataset of 2000 training sequences. The main files available are the following ones

    - Xtr.csv - the training sequences.
    - Xte.csv - the test sequences.
    - Ytr.csv - the sequence labels of the training sequences indicating bound (1) or not (0).

Each row of Xtr.csv represents a sequence. Xte.csv contains 1000 test sequences, for which you need to predict. Ytr.csv contains the labels corresponding to the training data, in the same format as a submission file.

### Optional Files
Besides these basic data files, we also provide some additional but not necessary data for those who prefer to work directly with numeric data.

    - Xtr_mat100.csv - the training feature matrices of size 2000 x 100.
    - Xte_mat100.csv - the test feature matrices of size 1000 x 100.

These feature matrices are calculated respectively from Xtr.csv and Xte.csv based on bag of words representation. Specifically, all the subsequences of length l (here l=10) are extracted from the sequences and are represented as a vector of 4xl dimensions using one-hot encoding (with A=(1, 0, 0, 0), C=(0, 1, 0, 0), G=(0, 0, 1, 0), T=(0, 0, 0, 1)). For example, if l=3 then ACA is represented as (1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0). Then, they are clustered into 50 clusters using Kmeans and each subsequence is assigned to a label i and is represented by a binary vector whose coefficients are equal to 0 except the ith one, which is equal to 1. Finally, for each sequence, we compute the average of the representations of all its subsequences to obtain the feature vector of this sequence.

The provided representations are not guaranteed to be optimal. If you want to obtain better performance, please work with the sequences directly.

## Evaluation 

### Submission Format
You must submit a csv file that contains two columns: **Id** and **Bound**. The file should contain a header and have the format described below. Id represents the number of the test example, ranging from 0 to 999. Bound is the corresponding label, either 0 or 1. 

The performance measure is the classification accuracy.

Example submission file:

```bash
Id,Bound
0,0
1,0
2,0
....
998,0
999,0
```