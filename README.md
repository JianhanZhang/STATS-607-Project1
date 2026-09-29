# An Evaluation of Matrix Factorization-based Methods for the Netflix Prize Dataset Prediction

## Project Introduction

### Background

Recommender systems have become an essential component in modern digital platforms, significantly enhancing user’s personalized experience across e-commerce (e.g., Amazon) and streaming services (e.g., Netflix). In 2006, the launch of the renowned Netflix Prize competition further sparked interest in this field by challenging researchers to elaborate on the accuracy of Netflix’s recommender system. 

At the core of a recommender system lies the task of predicting user ratings for unseen movies, which draws most of our attention. Given a dataset of known movie ratings provided by various users, the goal of this study is to predict unknown user ratings, enabling tailored recommendations that align with user preferences. Specifically, the input to our system is a massive sparse matrix of user-item ratings, and the output is a predicted rating for any given user-item pair.

To investigate this problem, we implement three matrix factorization-based methods for collaborative filtering: Biased Regularized Incremental Symmetric Matrix Factorization (BRISMF) [4], Iterative SVD [1], and SVD++ [3]. We then run them on a portion of the Netflix Prize dataset to evaluate and compare their performance. The results of the investigation are reported using tables and plots.

*(The above paragraphs are adapted from the project report submitted for the original project.)*

### Key Analysis

We conduct two key analyses of the models:

1. For each of the three models, we select their best hyperparameters using grid search. Then we fit the model using the selected hyperparameters on the training set and compute the resulting RMSE on the testing set. This analysis aims to evaluate the performance of the "best" version of each of the three models, within the limits of the hyperparameters attempted in the grid search.

2. For BRISMF and SVD++, we plot how the resulting RMSE varies with each of a subset of the hyperparameters. This analysis aims to explore how hyperparameter values affect the performance of BRISMF and SVD++.

### Expected Outputs

The first analysis produces a table containing the RMSEs of the three models under their respective selected hyperparameters. This table can be found in ``results/rmse_results.csv``.

The second analysis produces two plots showing how the RMSE varies with different hyperparameter values, one plot for each of BRISMF and SVD++. These plots can be found in ``results/brismf_rmse_analysis_plot.png`` (for BRISMF) and ``results/svdpp_rmse_analysis_plot.png`` (for SVD++).

## Project Structure

The project contains five major folders:

- **data/** Contains the Netflix Prize data used in the analysis.
- **originals/** Contains the Jupyter notebooks for the original (unorganized) analysis code.
- **results/** Contains the tables and plots resulting from the analysis. 
- **src/** Contains the source code of the project, including implementation of BRISMF and iterative SVD, as well as code to run the analysis. We do not implement SVD++; instead, we use the SVD++ implementation in the ``surprise`` library [2].
- **test/** Contains testing code.

```text
.
├── .gitignore
├── Makefile
├── README.md
├── data
│   └── data.csv
├── originals
│   ├── brismf.ipynb
│   ├── iterative_svd.ipynb
│   └── svdpp.ipynb
├── requirements.txt
├── results
│   ├── brismf_rmse_analysis_plot.png
│   ├── rmse_results.csv
│   └── svdpp_rmse_analysis_plot.png
├── src
│   ├── analysis.py
│   ├── evaluation.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── brismf.py
│   │   └── iterative_svd.py
│   └── run_analysis.py
└── test
    ├── test_prediction_output.py
    └── test_rmse_calculation.py
```

## Data Access

The original Netflix Prize dataset is public and can be accessed [here](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data/data).

The dataset used in this analysis is a processed version of the original Netflix Prize dataset, and it can be found at ``data/data.csv``.

## Setup

This project was developed using Python 3.14.7.

To set up the project, first clone the repository and move into the project directory.

```bash
git clone https://github.com/JianhanZhang/STATS-607-Project1.git
cd STATS-607-Project1
```

Once in the project directory, run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Tests

The test suite can be run with

```bash
make test
```

## Reproducing the Analysis

The following command reproduces the analysis

```bash
make reproduce
```

In particular, running `make reproduce` executes the full analysis without interactive input
and regenerates the following files:

- `results/rmse_results.csv`
- `results/brismf_rmse_analysis_plot.png`
- `results/svdpp_rmse_analysis_plot.png`

## References

[1] Ghazanfar, Mustansar Ali, and Adam Prügel-Bennett. "The Advantage of Careful Imputation Sources in Sparse Data-Environment of Recommender Systems: Generating Improved SVD-based Recommendations." Informatica (Slovenia) 37.1 (2013): 61-92.
[2] Hug, Nicolas. "Surprise: A Python library for recommender systems." Journal of Open Source Software 5.52 (2020): 2174.
[3] Koren, Yehuda. "Factorization meets the neighborhood: a multifaceted collaborative filtering model." Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 2008.
[4] Takács, Gábor, et al. "Matrix factorization and neighbor based algorithms for the netflix prize problem." Proceedings of the 2008 ACM conference on Recommender systems. 2008.