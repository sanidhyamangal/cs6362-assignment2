import numpy as np
import itertools

# This class will handle model selection: really simple for now, just n-fold cross validation
class CVModelSelection:
    # --- Constructor: X and y are the full datasets (features and targets), while `n_folds` indicates the number of folds to setup
    def __init__(self,X,y,n_folds):
        self.X = X
        self.y = y
        self.d,self.n = self.X.shape
        self.folds_ = self.generate_folds(n_folds)

    #

    # --- generate `n_folds` number of folds
    def generate_folds(self,n_folds):
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
    def grid_search(self,Kernel,all_params):
        print("Grid Search Commended on {} for kernel".format(all_params))
        self.accuracy_mesh_ = np.empty(shape=[len(all_params['length_scales']), len(all_params['noise_variance'])])
        
        for idx_l_scale,length_scale in enumerate(all_params['length_scales']):
            for idx_noise_variance, noise_variance in enumerate(all_params['noise_variance']):
                print(f"Checking for following hyperpram sets, length_scale:{length_scale}, noise_variance:{noise_variance}")
                _test_fold = np.random.randint(0,len(self.folds_), 1)[0]
                _param_accuracy = []
                for fold in range(len(self.folds_)):
                    if fold == _test_fold:
                        continue
                    kernel = Kernel(self.X[self.folds_[fold]], self.y[self.folds_[fold]], length_scale, noise_variance)
                    _,_predictions = kernel.sample_from_gp(self.X[self.folds_[_test_fold]], n_draws=1)
                    _predictions = _predictions.reshape(-1)
                    _param_accuracy.append(np.square(self.y[self.folds_[fold]]-_predictions).mean())
            self.accuracy_mesh_[idx_l_scale, idx_noise_variance] = np.mean(_param_accuracy)

        _best_params = np.argwhere(self.accuracy_mesh_ == np.min(self.accuracy_mesh_))

        return all_params['length_scales'][_best_params[0][0]],all_params['noise_variance'][_best_params[0][1]]

    #
#
