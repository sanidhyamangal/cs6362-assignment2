import numpy as np
import itertools

# This class will handle model selection: really simple for now, just n-fold cross validation
class CVModelSelection:
    # --- Constructor: X and y are the full datasets (features and targets), while `n_folds` indicates the number of folds to setup
    def __init__(self,X,y,n_folds):
        self.X = X
        self.y = y
        self.d,self.n = self.X.shape
    #

    # --- generate `n_folds` number of folds
    def generate_folds(self,n_folds):
        pass
    #

    # --- perform grid search: assumes folds exist! This method should return the best set of hyperparameters: use mean-squared error wrt withheld data
    #   (1) the Kernel argument is a class, and should be used to construct an instance of a kernel
    #   (2) all_params is a list where each item corresponds to a single hyperparameter, and contains a range of values for the hyperparameter
    def grid_search(self,Kernel,all_params):
        pass
    #
#
