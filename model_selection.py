import numpy as np
import itertools


# This class will handle model selection: really simple for now, just n-fold cross validation
class CVModelSelection:
    # --- Constructor: X and y are the full datasets (features and targets), while `n_folds` indicates the number of folds to setup
    def __init__(self, X, y, n_folds):
        self.X = X
        self.y = y
        self.d, self.n = self.X.shape
        self.folds_ = self.generate_folds(n_folds)

    #

    # --- generate `n_folds` number of folds
    def generate_folds(self, n_folds):
        # create a range array for the shape of data
        _range_array = np.arange(0, stop=self.d, step=1)

        # first shuffle the range array and then split it into n_folds using split method
        np.random.shuffle(_range_array)

        # return split indices of array which can later be used for slicing X and y
        return np.array_split(_range_array, n_folds)

    #

    # --- perform grid search: assumes folds exist! This method should return the best set of hyperparameters: use mean-squared error wrt withheld data
    #   (1) the Kernel argument is a class, and should be used to construct an instance of a kernel
    #   (2) all_params is a list where each item corresponds to a single hyperparameter, and contains a range of values for the hyperparameter
    def grid_search(self, Kernel, all_params):
        # a message for indicating gridsearch is commended on params
        print("Grid Search Commenced on {} for kernel".format(all_params))

        # calculate accuracy_mesh to store all the _mse for a grid
        self.accuracy_mesh_ = np.empty(shape=[
            len(all_params['length_scales']),
            len(all_params['noise_variance'])
        ])

        # iterate over all length_scale and noise_variance in nested fashion
        for idx_l_scale, length_scale in enumerate(
                all_params['length_scales']):
            for idx_noise_variance, noise_variance in enumerate(
                    all_params['noise_variance']):

                # _generate test_fold from n_folds
                _test_fold = np.random.randint(0, len(self.folds_), 1)[0]

                # init an empty array for storing all the mse and later which would be meaned for the entire folds
                _mse = []
                # iterate over all the folds for the eval op
                for fold in range(len(self.folds_)):

                    # if current fold is test_fold skip the ops
                    if fold == _test_fold:
                        continue

                    # init a kernel with the fold data
                    kernel = Kernel(self.X[self.folds_[fold]],
                                    self.y[self.folds_[fold]], length_scale,
                                    noise_variance)

                    # draw means and sample from the kernel
                    _, _predictions = kernel.sample_from_gp(
                        self.X[self.folds_[_test_fold]], n_draws=1)

                    # reshape predictions into [n,] for eval purpose
                    _predictions = _predictions.reshape(-1)
                    # append the mse to _mse vector to later compute mean for the current params
                    _mse.append(
                        np.square(self.y[self.folds_[fold]] -
                                  _predictions).mean())

                # store mean of _mse in accuracy mesh to compute the best_params later
                self.accuracy_mesh_[idx_l_scale,
                                    idx_noise_variance] = np.mean(_mse)

        # extrat the location of best_params from the accuracy mesh with min mse
        _best_params = np.argwhere(
            self.accuracy_mesh_ == np.min(self.accuracy_mesh_))

        # return best length_scale and noise_variance
        return all_params['length_scales'][_best_params[0][0]], all_params[
            'noise_variance'][_best_params[0][1]]

    #


#
