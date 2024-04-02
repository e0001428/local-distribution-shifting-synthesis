# Detecting Leaked Data through Synthetic Data Injection and Model Querying

Welcome to this GitHub!

In this repository, we develop a new algorithm, LDSS, to detect leaked data through synthetic data injection and model querying. Shifting local class distribution enables accurate detection of whether a classification model is trained using leaked tabular data. 

We devise LDSS with optimization using **[FAISS_GPU]** (Jeff Johnson, Matthijs Douze, and Herve Jegou. 2021. Billion-Scale Similarity
Search with GPUs. IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2021). We also include the implementation and comparison to two baseline methods, namely **Flip** and **FlipNN**, that perform random label flipping or random nearest neighbor flipping. 

## Data Sets

### Data Sets Details

We study the performance of LDSS on five real-world datasets, i.e., Adult, Vermont, Arizona, Covertype, and GeoNames. Their statistics after data cleaning are summarized as follows.

| Datasets  | Cardinality  | # Categorical Attributes | # Numerical Attributes | # Classes |   g   |   h   | Download Link |
| --------- | ----------   | ------------------------ | ---------------------- | --------- | ----- | ----- | ------------- |
| Adult     | 48,842       |      8                   |   5                    |   2       |  10   | 44    | <https://archive.ics.uci.edu/dataset/2/adult> |
| Vermont   | 129,816      |      40                  |   8                    |   4       |  10   | 118   | <https://datacatalog.urban.org/dataset/2018-differential-privacy-synthetic-data-challenge-datasets/resource/2478d8a8-1047-451b-ae23> |
| Arizona   | 203,353      |      42                  |   8                    |   4       |  10   | 185   | <https://datacatalog.urban.org/dataset/2018-differential-privacy-synthetic-data-challenge-datasets/resource/2478d8a8-1047-451b-ae23> |
| Covertype | 581,012      |      2                   |  10                    |   7       |  10   | 528   | <https://archive.ics.uci.edu/dataset/31/covertype> |
| GeoNames  | 1,891,513    |      2                   |   5                    |   9       |  10   | 1720  | <https://www.kaggle.com/datasets/geonames/geonames-database> |


### Data Format

Input data is in CSV format. The first row corresponds to column names. The second row corresponds to column data types: 'C' for categorical features, 'R' for real-valued numerical features, and 'I' for integer features. The remaining rows are data samples with 1 sample per row. The last column is always the class label column with column data type 'C'. Refer to `data/adult/adult.data` for the Adult dataset and `data/arizona/arizona.data` for the Arizona dataset as examples.


## Requirements

- Ubuntu 20.04 (or higher version)
- Python 3.10 (or higher version)
- GPU with CUDNN available for TensorFlow 2.11 and FAISS_GPU 1.7.2 (or higher version)
- Python libraries listed in requirements.txt


## Experiments

We provide the bash scripts to run all experiments. Once you have downloaded the data sets, you can reproduce the experiments by simply running the following commands:

```bash
cd models
bash scripts/run.sh
```
For the respective dataset's experiment script, `test\_{dataset}.sh` runs **LDSS** experiments, `test\_{dataset}\_b.sh` runs **Flip and FlipNN** experiments, `test\_{dataset}\_gh.sh` runs **parameter study** experiments, and `test\_{dataset}\_r.sh` runs **regression** experiments. 

To run experiments on customized datasets, please run the below command with the right arguments provided.

```bash
python model/main.py **args
```

Some useful arguments are listed below.
|  Parameter         | Description |
| ---------          | ----------  |
|  --dataset         | {adult,covertype,arizona,vermont,geonames}, dataset name with respective data file at data/dataset/\*.data |
|  --result-dir      | result dir |
|  --seed SEED       | random number seed |
|  --data-perhole    | h, the number of samples to be generated per empty ball |
|  --num-holes       | g, the number of empty balls to find and inject data |
|  --test-config     | path of testing configuration file |
|  --contamination   | contamination for iso forest |
|  --restore RESTORE | whether add random data to restore original class distribution |
|  --split-ratio     | k, fold for k-fold experiments |
|  --split-portion   | which portion is used as training data (0 to k-1), if split ratio is k-fold |
|  --npivot          | m, the number of pivot samples in jacaard distance calculation |
|  --nbit NBIT       | k, the number of bits per feature in jacaard distance calculation |
|  --pivot-method    | {random,maxfreq,kpp} method of pivot selection |
|  --regression      | {T,F} whether to enable regression task, available for Arizona and Vermont for Jaccard method |
|  --eval-multiplier | dilution multiplier to be tested from 1 to k-1 on k-fold (only up to 1 for regression) |


## Evaluations

Finally, we provide sample Python codes in Jupyter Notebook for evaluations and case studies. Please refer to the folders `models/Evaluation.ipynb` for more details. To evaluate the experiment results, replace the result file path accordingly.
