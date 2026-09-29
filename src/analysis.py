'''Implement functions for analyzing the performance of BRISMF, iterative SVD, and SVD++ for collaborative filtering.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.brismf import BRISMF
from models.iterative_svd import IterativeSVD
from evaluation import compute_rmse
from surprise import reader, Dataset, Trainset, SVDpp, accuracy
from surprise.model_selection import train_test_split
from surprise.dataset import DatasetAutoFolds


def best_rmse_brismf(
        train_data, 
        non_validation_data, 
        validation_data, 
        test_data, 
        K_list=[10, 100, 1000], 
        lr_p_list=[0.00003, 0.0003], 
        lr_q_list=[0.00003, 0.0003], 
        p_p_list=[0.001, 0.01], 
        p_q_list=[0.001, 0.01]
):
    """Select and evaluate the best BRISMF model.

    First select the best parameters for BRISMF using grid search on the validation set, and then evaluate and  
    return the RMSE of the selected model on the test set.

    Parameters
    ----------
    train_data : pandas.DataFrame
        Full training dataset.
    non_validation_data : pandas.DataFrame
        Subset of the training data used to train models during hyperparameter selection.
    validation_data : pandas.DataFrame
        Validation data used to compute the RMSE during hyperparameter selection.
    test_data : pandas.DataFrame
        Test data used for final RMSE evaluation of the selected model.
    K_list : list of int, default=[10, 100, 1000]
        Candidate latent factor dimensions.
    lr_p_list : list of float, default=[0.00003, 0.0003]
        Candidate learning rates for user latent factors.
    lr_q_list : list of float, default=[0.00003, 0.0003]
        Candidate learning rates for movie latent factors.
    p_p_list : list of float, default=[0.001, 0.01]
        Candidate regularization penalties for user latent factors.
    p_q_list : list of float, default=[0.001, 0.01]
        Candidate regularization penalties for movie latent factors.

    Returns
    -------
    float
        Test RMSE of the selected BRISMF model."""
    
    # Find the best hyperparameters for BRISMF using grid search
    best_rmse = float('inf')
    best_params = []
    for K in K_list:
        for lr_p in lr_p_list:
            for lr_q in lr_q_list:
                for p_p in p_p_list:
                    for p_q in p_q_list:
                        model = BRISMF(K=K, lr_p=lr_p, lr_q=lr_q, p_p=p_p, p_q=p_q)
                        model.fit(non_validation_data.drop(columns=['rating']), non_validation_data['rating'])
                        rmse = compute_rmse(validation_data['rating'].to_numpy(), model.predict(validation_data))
                        if (rmse < best_rmse):
                            best_rmse = rmse
                            best_params = [K, lr_p, lr_q, p_p, p_q]

    # Fit BRISMF using the best hyperparameters selected from the grid search
    brismf_100000 = BRISMF(K=best_params[0], epochs=10, lr_p=best_params[1], lr_q=best_params[2], p_p=best_params[3], p_q=best_params[4])
    brismf_100000.fit(train_data.drop(columns=['rating']), train_data['rating'])

    # Compute and return the resulting RMSE
    return compute_rmse(test_data['rating'].to_numpy(), brismf_100000.predict(test_data))


def best_rmse_iterative_svd(
        train_data, 
        non_validation_data, 
        validation_data, 
        test_data, 
        K_list=[10, 20, 100], 
):
    """Select and evaluate the best IterativeSVD model.

    First select the best parameters for IterativeSVD using grid search on the validation set, and then evaluate and  
    return the RMSE of the selected model on the test set.

    Parameters
    ----------
    train_data : pandas.DataFrame
        Full training dataset.
    non_validation_data : pandas.DataFrame
        Subset of the training data used to train models during hyperparameter selection.
    validation_data : pandas.DataFrame
        Validation data used to compute the RMSE during hyperparameter selection.
    test_data : pandas.DataFrame
        Test data used for final RMSE evaluation of the selected model.
    K_list : list of int, default=[10, 20, 100]
        Candidate latent factor dimensions.

    Returns
    -------
    float
        Test RMSE of the selected IterativeSVD model.
    """
    
    # Find the best hyperparameters for iterative SVD using grid search
    best_rmse = np.inf
    best_params = None
    for K in K_list:
        model = IterativeSVD(K, 3)
        model.fit(non_validation_data.drop(columns=['rating']), non_validation_data['rating'])
        rmse = compute_rmse(validation_data['rating'].to_numpy(), model.predict(validation_data))
        if (rmse < best_rmse):
            best_rmse = rmse
            best_params = [K]

    # Fit iterative SVD using the best hyperparameters selected from the grid search
    iter_svd_100000 = IterativeSVD(k=best_params[0], num_epochs=3)
    iter_svd_100000.fit(train_data[['movie_id', 'user_id']], train_data['rating'])

    # Compute and return the resulting RMSE
    return compute_rmse(test_data['rating'].to_numpy(), iter_svd_100000.predict(test_data))


def best_rmse_svdpp(
        train_data, 
        non_validation_data, 
        validation_data, 
        test_data, 
        n_factors_list = [10, 20, 100], 
        n_epochs_list = [10, 20, 100], 
        lr_all_list = [0.002, 0.02], 
        reg_all_list = [0.02, 0.2]
):
    """Select and evaluate the best SVD++ model.

    First select the best parameters for SVD++ using grid search on the validation set, and then evaluate and  
    return the RMSE of the selected model on the test set.

    Parameters
    ----------
    train_data : pandas.DataFrame
        Full training dataset.
    non_validation_data : pandas.DataFrame
        Subset of the training data used to train models during hyperparameter selection.
    validation_data : pandas.DataFrame
        Validation data used to compute the RMSE during hyperparameter selection.
    test_data : pandas.DataFrame
        Test data used for final RMSE evaluation of the selected model.
    n_factors_list : list of int, default=[10, 20, 100]
        Candidate numbers of latent factors.
    n_epochs_list : list of int, default=[10, 20, 100]
        Candidate numbers of training epochs.
    lr_all_list : list of float, default=[0.002, 0.02]
        Candidate learning rates.
    reg_all_list : list of float, default=[0.02, 0.2]
        Candidate regularization penalties.

    Returns
    -------
    float
        Test RMSE of the selected SVD++ model."""

    # Load data into required format for the surprise library
    ratings_reader = reader.Reader(rating_scale=(1,5))
    train_dset = Dataset.load_from_df(train_data[['user_id', 'movie_id', 'rating']], ratings_reader)
    test_dset = Dataset.load_from_df(test_data[['user_id', 'movie_id', 'rating']], ratings_reader)
    trainset = DatasetAutoFolds.build_full_trainset(train_dset)
    testset = [tuple(x) for x in test_data[['user_id', 'movie_id', 'rating']].values]
    train_dset_non_valid = Dataset.load_from_df(non_validation_data[['user_id', 'movie_id', 'rating']], ratings_reader)
    test_dset_valid = Dataset.load_from_df(validation_data[['user_id', 'movie_id', 'rating']], ratings_reader)
    trainset_non_valid = DatasetAutoFolds.build_full_trainset(train_dset_non_valid)
    testset_valid = [tuple(x) for x in validation_data[['user_id', 'movie_id', 'rating']].values]

    # Find the best hyperparameters for SVD++ using grid search
    best_rmse = np.inf
    best_params = None
    for n_factors in n_factors_list:
        for n_epochs in n_epochs_list:
            for lr_all in lr_all_list:
                for reg_all in reg_all_list:
                    model = SVDpp(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
                    model.fit(trainset_non_valid)
                    predictions = model.test(testset_valid)
                    rmse = accuracy.rmse(predictions)
                    print(n_factors, n_epochs, lr_all, reg_all)
                    if (rmse < best_rmse):
                        best_rmse = rmse
                    best_params = [n_factors, n_epochs, lr_all, reg_all]

    # Fit SVD++ using the best hyperparameters selected from the grid search
    svdpp = SVDpp(n_factors=best_params[0], n_epochs=best_params[1], lr_all=best_params[2], reg_all=best_params[3])
    svdpp.fit(trainset)
    predictions = svdpp.test(testset)
    
    # Compute and return the resulting RMSE
    return accuracy.rmse(predictions)


def best_rmse_all(
        train_data, 
        non_validation_data, 
        validation_data, 
        test_data, 
        K_list_brismf=[10, 100, 1000], 
        lr_p_list_brismf=[0.00003, 0.0003], 
        lr_q_list_brismf=[0.00003, 0.0003], 
        p_p_list_brismf=[0.001, 0.01], 
        p_q_list_brismf=[0.001, 0.01], 
        K_list_iterative_svd=[10, 20, 100], 
        n_factors_list_svdpp = [10, 20, 100], 
        n_epochs_list_svdpp = [10, 20, 100], 
        lr_all_list_svdpp=[0.002, 0.02], 
        reg_all_list_svdpp=[0.02, 0.2]
):
    """Evaluate all recommendation models and save their RMSE values.

    For each of BRISMF, Iterative SVD, and SVD++, first select the best parameters for SVD++ using grid search 
    on the validation set, and then evaluate the RMSE of the selected model on the test set. The resulting RMSE 
    values are saved to ``results/rmse_results.csv``.

    Parameters
    ----------
    train_data : pandas.DataFrame
        Full training dataset.
    non_validation_data : pandas.DataFrame
        Portion of the training data used to fit candidate models.
    validation_data : pandas.DataFrame
        Validation data used for hyperparameter selection.
    test_data : pandas.DataFrame
        Test data used for final evaluation.
    K_list_brismf : list of int, default=[10, 100, 1000]
        Candidate latent dimensions for BRISMF.
    lr_p_list_brismf : list of float, default=[0.00003, 0.0003]
        Candidate user factor learning rates for BRISMF.
    lr_q_list_brismf : list of float, default=[0.00003, 0.0003]
        Candidate movie factor learning rates for BRISMF.
    p_p_list_brismf : list of float, default=[0.001, 0.01]
        Candidate user factor penalties for BRISMF.
    p_q_list_brismf : list of float, default=[0.001, 0.01]
        Candidate movie factor penalties for BRISMF.
    K_list_iterative_svd : list of int, default=[10, 20, 100]
        Candidate latent dimensions for IterativeSVD.
    n_factors_list_svdpp : list of int, default=[10, 20, 100]
        Candidate numbers of latent factors for SVD++.
    n_epochs_list_svdpp : list of int, default=[10, 20, 100]
        Candidate numbers of training epochs for SVD++.
    lr_all_list_svdpp : list of float, default=[0.002, 0.02]
        Candidate learning rates for SVD++.
    reg_all_list_svdpp : list of float, default=[0.02, 0.2]
        Candidate regularization penalties for SVD++.
    """
    
    # Compute the best RMSE for each model
    rmse_brismf = best_rmse_brismf(train_data, non_validation_data, validation_data, test_data, K_list_brismf, lr_p_list_brismf, lr_q_list_brismf, p_p_list_brismf, p_q_list_brismf)
    rmse_iterative_svd = best_rmse_iterative_svd(train_data, non_validation_data, validation_data, test_data, K_list_iterative_svd)
    rmse_svdpp = best_rmse_svdpp(train_data, non_validation_data, validation_data, test_data, n_factors_list_svdpp, n_epochs_list_svdpp, lr_all_list_svdpp, reg_all_list_svdpp)

    # Save the RMSE results to a CSV file
    rmse_results = pd.DataFrame({
        'Model': ['BRISMF', 'Iterative SVD', 'SVD++'],
        'RMSE': [rmse_brismf, rmse_iterative_svd, rmse_svdpp]
    })
    rmse_results.to_csv('results/rmse_results.csv', index=False)


def plot_brismf_details(
        train_data, 
        test_data, 
        epochs_list = [10, 20, 50, 100], 
        k_list = [10, 20, 50, 100, 1000], 
        p_pb_list = [0.001, 0.005, 0.01, 0.05, 0.1], 
        p_qb_list = [0.001, 0.005, 0.01, 0.05, 0.1]
):
    """Plot BRISMF RMSE sensitivity to selected hyperparameters.

    Fit BRISMF models over several values of the number of epochs, latent
    dimension, user bias penalty, and movie bias penalty, then save the
    resulting RMSE plots. The resulting plot is saved to 
    ``results/brismf_rmse_analysis_plot.png``.


    Parameters
    ----------
    train_data : pandas.DataFrame
        Data used to fit each BRISMF model.
    test_data : pandas.DataFrame
        Data used to evaluate the RMSE of each BRISMF model.
    epochs_list : list of int, default=[10, 20, 50, 100]
        List of numbers of training epochs to be evaluated.
    k_list : list of int, default=[10, 20, 50, 100, 1000]
        List of latent dimensions to be evaluated.
    p_pb_list : list of float, default=[0.001, 0.005, 0.01, 0.05, 0.1]
        List of user bias regularization penalties to be evaluated.
    p_qb_list : list of float, default=[0.001, 0.005, 0.01, 0.05, 0.1]
        List of movie bias regularization penalties to be evaluated.
    """

    # Local helper function to compute RMSE for a given list of hyperparameters of interest
    def batch_rmse(train_set, test_set, K=[10], p_p=[0.01], p_pb=[0.01], p_q=[0.01], p_qb=[0.01], epochs=[10]):
        rmse_list = []
        for k in K:
            for pp in p_p:
                for ppb in p_pb:
                    for pq in p_q:
                        for pqb in p_qb:
                            for e in epochs:
                                model = BRISMF(K=k, p_p=pp, p_pb=ppb, p_q=pq, p_qb=pqb, epochs=e, lr_p=0.0003, lr_q=3e-05)
                                model.fit(train_set.drop(columns=['rating']), train_set['rating'])
                                rmse_list.append(compute_rmse(test_set['rating'], model.predict(test_set.drop(columns=['rating']))))

        return np.array(rmse_list)

    # Compute batch RMSE for each of the hyperparameter lists of interest
    rmse_list_epochs = batch_rmse(train_data, test_data, epochs=epochs_list)
    rmse_list_k = batch_rmse(train_data, test_data, K=k_list)
    rmse_list_p_pb = batch_rmse(train_data, test_data, p_pb=p_pb_list)
    rmse_list_p_qb = batch_rmse(train_data, test_data, p_qb=p_qb_list)

    # Plot the RMSE results for each of the hyperparameter lists of interest
    # Create a grid of subplots
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # First plot
    x1 = epochs_list
    axes[0].plot(x1, rmse_list_epochs, label='Test RMSE', color='blue', marker='o', linewidth=2)
    axes[0].set_title('Test RMSE by Number of Epochs', fontsize=14)
    axes[0].set_xlabel('Number of Epochs', fontsize=12)
    axes[0].set_ylabel('Test RMSE', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)

    # Second plot
    x2 = k_list
    axes[1].plot(x2, rmse_list_k, label='Test RMSE', color='red', marker='o', linewidth=2)
    axes[1].set_title('Test RMSE by Number of Dimensions (K)', fontsize=14)
    axes[1].set_xlabel('Number of Dimensions (K)', fontsize=12)
    axes[1].set_ylabel('Test RMSE', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)

    # Third plot
    x3 = p_pb_list
    axes[2].plot(x3, rmse_list_p_pb, label='Test RMSE', color='green', marker='o', linewidth=2)
    axes[2].set_title('Test RMSE by User Bias Penalty (ppb)', fontsize=14)
    axes[2].set_xlabel('User Bias Penalty (ppb)', fontsize=12)
    axes[2].set_ylabel('Test RMSE', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(fontsize=10)

    # Fourth plot
    x4 = p_qb_list
    axes[3].plot(x4, rmse_list_p_qb, label='Test RMSE', color='green', marker='o', linewidth=2)
    axes[3].set_title('Test RMSE by Movie Bias Penalty (pqb)', fontsize=14)
    axes[3].set_xlabel('Movie Bias Penalty (pqb)', fontsize=12)
    axes[3].set_ylabel('Test RMSE', fontsize=12)
    axes[3].grid(True, linestyle='--', alpha=0.6)
    axes[3].legend(fontsize=10)

    # Adjust layout of the plots
    plt.tight_layout()

    # Add a title for the entire figure
    fig.suptitle('Analysis of Test RMSE for BRISMF', fontsize=16, y=1.02)

    # Save the figure as a PNG file
    plt.savefig('results/brismf_rmse_analysis_plot.png', bbox_inches='tight')


def plot_svdpp_details(
        train_data, 
        test_data, 
        n_epochs_list=[10,20,50,100], 
        n_factors_list=[10,20,50,100,1000], 
        lr_all_list=[0.007, 0.02, 0.07, 0.2], 
        reg_all_list=[0.0002, 0.007, 0.02, 0.07, 0.2]
):
    """Plot SVD++ RMSE sensitivity to selected hyperparameters.

    Fit SVD++ models over several values of the number of epochs, latent
    dimension, learning rate, regularization penalty, then save the
    resulting RMSE plots. The resulting plot is saved to 
    ``results/svdpp_rmse_analysis_plot.png``.


    Parameters
    ----------
    train_data : pandas.DataFrame
        Data used to fit each SVD++ model.
    test_data : pandas.DataFrame
        Data used to evaluate the RMSE of each SVD++ model.
    n_epochs_list : list of int, default=[10, 20, 50, 100]
        List of numbers of training epochs to be evaluated.
    n_factors_list : list of int, default=[10, 20, 50, 100, 1000]
        List of numbers of latent factors to be evaluated
    lr_all_list : list of float, default=[0.007, 0.02, 0.07, 0.2]
        List of learning rates to evaluate.
    reg_all_list : list of float, default=[0.0002, 0.007, 0.02, 0.07, 0.2]
        List of regularization penalties to evaluate.
    """

    # Load data into required format for the surprise library
    ratings_reader = reader.Reader(rating_scale=(1,5))
    train_dset = Dataset.load_from_df(train_data[['user_id', 'movie_id', 'rating']], ratings_reader)
    test_dset = Dataset.load_from_df(test_data[['user_id', 'movie_id', 'rating']], ratings_reader)
    trainset = DatasetAutoFolds.build_full_trainset(train_dset)
    testset = [tuple(x) for x in test_data[['user_id', 'movie_id', 'rating']].values]

    # Local helper function to compute RMSE for a given list of hyperparameters of interest
    def svdpp_batch_rmse(trainset, testset, n_factors=[10], n_epochs=[10], lr_all =[0.02], reg_all=[0.2]):
        rmse_list = []
        for k in n_factors:
            for e in n_epochs:
                for r in reg_all:
                    for l in lr_all:
                        model = SVDpp(n_factors=k, n_epochs=e, lr_all=l, reg_all=r)
                        model.fit(trainset)
                        predictions = model.test(testset)
                        rmse_list.append(accuracy.rmse(predictions))
        return np.array(rmse_list)

    # Compute batch RMSE for each of the hyperparameter lists of interest
    rmse_list_n_epochs = svdpp_batch_rmse(trainset, testset, n_epochs=n_epochs_list)
    rmse_list_n_factors = svdpp_batch_rmse(trainset, testset, n_factors=n_factors_list)
    rmse_list_lr_all = svdpp_batch_rmse(trainset, testset, lr_all=lr_all_list)
    rmse_list_reg_all = svdpp_batch_rmse(trainset, testset, reg_all=reg_all_list)

    # Plot the RMSE results for each of the hyperparameter lists of interest
    # Create a grid of subplots
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # First plot
    x1 = n_epochs_list
    axes[0].plot(x1, rmse_list_n_epochs, label='Test RMSE', color='blue', marker='o', linewidth=2)
    axes[0].set_title('Test RMSE by Number of Epochs', fontsize=14)
    axes[0].set_xlabel('Number of Epochs', fontsize=12)
    axes[0].set_ylabel('Test RMSE', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)

    # Second plot
    x2 = n_factors_list
    axes[1].plot(x2, rmse_list_n_factors, label='Test RMSE', color='red', marker='o', linewidth=2)
    axes[1].set_title('Test RMSE by Number of Factors (n_factors)', fontsize=14)
    axes[1].set_xlabel('Number of Factors (n_factors)', fontsize=12)
    axes[1].set_ylabel('Test RMSE', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)

    # Third plot
    x3 = lr_all_list
    axes[2].plot(x3, rmse_list_lr_all, label='Test RMSE', color='green', marker='o', linewidth=2)
    axes[2].set_title('Test RMSE by Learning Rate (lr_all)', fontsize=14)
    axes[2].set_xlabel('Learning Rate (lr_all)', fontsize=12)
    axes[2].set_ylabel('Test RMSE', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(fontsize=10)

    # Fourth plot
    x4 = reg_all_list
    axes[3].plot(x4, rmse_list_reg_all, label='Test RMSE', color='orange', marker='o', linewidth=2)
    axes[3].set_title('Test RMSE by Regularization Penalty (reg_all)', fontsize=14)
    axes[3].set_xlabel('Regularization Penalty (reg_all)', fontsize=12)
    axes[3].set_ylabel('Test RMSE', fontsize=12)
    axes[3].grid(True, linestyle='--', alpha=0.6)
    axes[3].legend(fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Add a title for the entire figure
    fig.suptitle('Analysis of Test RMSE for SVD++', fontsize=16, y=1.02)

    # Save the figure as a PNG file
    plt.savefig('results/svdpp_rmse_analysis_plot.png', bbox_inches='tight')