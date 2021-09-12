import numpy as np
import itertools

from kernels import SquaredExponentialKernel
from model_selection import CVModelSelection

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_mean_and_draws(mean, draws):
    fig = plt.figure(figsize=(4, 4))
    image_grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.1)
    min_val = min(np.min(mean), np.min(draws))
    max_val = min(np.max(mean), np.max(draws))

    centered_draws = [draw for draw in draws[:4]
                      ] + [mean] + [draw for draw in draws[4:]]
    for ax, draw in zip(image_grid, centered_draws):
        ax.imshow(draw, cmap='Spectral', vmin=min_val, vmax=max_val)
        ax.autoscale(False)
        ax.axis('off')
    #
    plt.show()


#


def evaluate_test(predicted):
    pass


#

if __name__ == '__main__':
    X = np.load('data/X_train.npy')
    y = np.load('data/y_train.npy')

    # In this main you should:
    # (1) --- construct cross validation model selector
    # define GridSearchCV on 5 folds
    model_selector = CVModelSelection(X.T, y, 5)

    # (2) --- define hyperparameter ranges for grid search
    # define log spaced hyperprams for the grid search cv
    hyper_params = {
        "length_scales": np.geomspace(0.001, 0.01, num=10),
        "noise_variance": np.geomspace(0.001, 5, num=5)
    }

    # (3) --- run grid search: return best hyperparameters
    # and return the best length_scale and noise variance
    length_scale, noise_variance = model_selector.grid_search(
        SquaredExponentialKernel, hyper_params)

    # (4) construct kernel on full dataset (X,y) from above found hyperparameters, compute posterior mean from X_test, call `evaluate_test` function above with mean
    X_test = np.load('data/X_test.npy')

    # construct SQE kernel with best params
    kernel = SquaredExponentialKernel(X.T, y, length_scale, noise_variance)

    # compute the mean and predictions on the test data
    _mean, _predictions = kernel.sample_from_gp(X_test.T)

    # evaluate the predictions
    # reshape the predictions in [n,] shape.
    evaluate_test(_predictions.reshape(-1))

    # --- uncomment this out once you have your code working to see random draws (and mean) from the posterior
    # ---> the below assumes the presence of a `kernel` variable

    # res = 64
    # x_samples = np.linspace(-.9,.9,res)
    # y_samples = np.linspace(-.9,.9,res)

    # x_grid,y_grid = np.meshgrid(x_samples,y_samples,indexing='ij')
    # main_grid = np.stack((x_grid,y_grid),axis=0)

    # field_mean,field_draws = kernel.sample_from_gp(main_grid.reshape(2,-1).T,n_draws=8)
    # field_mean = field_mean.reshape(res,res)
    # field_draws = field_draws.reshape(8,res,res)
    # plot_mean_and_draws(field_mean,field_draws)

#
