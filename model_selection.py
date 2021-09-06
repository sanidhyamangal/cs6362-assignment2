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
        # using vlsplit to split the data in n_folds
        _X_splits = np.vsplit(self.X, n_folds)

        # since data is only is 1-D in Y hence, to apply vstack we need to add a simple axis X
        # now once the vtsack is applied we revert the shape of splits to 1-D space
        _y_splits = list(map(lambda x: x.reshape(-1), np.vsplit(np.reshape(self.y, newshape=[-1,1]),5)))

        # return stacked data (X_splits and y_splits)
        return zip(_X_splits, _y_splits)

    #

    # --- perform grid search: assumes folds exist! This method should return the best set of hyperparameters: use mean-squared error wrt withheld data
    #   (1) the Kernel argument is a class, and should be used to construct an instance of a kernel
    #   (2) all_params is a list where each item corresponds to a single hyperparameter, and contains a range of values for the hyperparameter
    def grid_search(self,Kernel,all_params):
        pass
    #
#
