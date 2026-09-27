import numpy as np
import pandas as pd
import random
from analysis import best_rmse_all, plot_brismf_details, plot_svdpp_details

# Set random seed for reproducibility
np.random.seed(1)
random.seed(1)

# We split the data temporally. In other words, we first find the 100000 most recent ratings. 
# We then reserve the newest 20000 of the 100000 ratings as the test set, while the remaining 80000 is 
# used as the training set.
full = pd.read_csv('../data/data.csv').drop(columns=['Unnamed: 0'])[-100000:]
train = full[:80000]
test = full[-20000:]

# We use the validation set approach to select the hyperparameters for each of the models. 
# In this case, we further divide the training set for the models into a non-validation set of 
# size 64000 (used for training in the validation set approach) and a validation set of 
# size 16000 (used for testing in the validation set approach).
non_validation = train[:64000]
validation = train[64000:]

# ANALYSIS 1: For each of the algorithms, we use grid search to select the best hyperparameters, and then use 
# the best hyperparameters to train the model. We save the RMSEs from the algorithms into a CSV file.
best_rmse_all(
    train_data=train, 
    non_validation_data=non_validation, 
    validation_data=validation, 
    test_data=test, 
)

# ANALYSIS 2: For BRISMF and SVD++, we run the algorithms with different hyperparameters, 
# and then plot the RMSEs to analyze the impact of each hyperparameter on the RMSE.
plot_brismf_details(train_data=train, test_data=test)
plot_svdpp_details(train_data=train, test_data=test)