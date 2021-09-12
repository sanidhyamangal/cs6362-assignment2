import numpy as np
from scipy.linalg import cho_factor, cho_solve


# Base class for a Gaussian Process kernel
class Kernel:
    # --- Constructor: in here, you _should_:
    #   (1) create the kernel matrix for the given data -- X, a (d x n) matrix
    #   (2) compute, and store, its Cholesky factorization, and any other relevant information for computing the mean, and covariance, in methods below
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.d, self.n = self.X.shape
        self.K = None

    #

    # --- form, and return, the posterior mean, conditioned on X_star: a set of data points
    def form_posterior_mean(self, X_star):
        # compute K_star
        K_star = self.kernel(self.X, X_star)

        # compute mean
        _mean = K_star.T @ np.linalg.inv(self.K) @ self.y

        return _mean

    #

    # --- form, and return, the posterior covariance, conditioned on X_star
    def form_posterior_covariance(self, X_star):
        # compute K_** and K_*
        K_star_star = self.kernel(X_star, X_star)
        K_star = self.kernel(self.X, X_star)

        # compute cov using K_** - K_*^TK^-1K_star
        _cov = K_star_star - K_star.T @ np.linalg.inv(self.K) @ K_star

        # return cov matrix
        return _cov

    #

    # --- draw samples from the posterior, conditioned on X_star
    #   (1) n_draws is a positive integer, indicating the number of draws to make
    #   (2) epsilon is a small number to ensure the posterior covariance is not ill-conditioned
    def sample_from_gp(self, X_star, n_draws=1, epsilon=1e-8):
        # compute posterior mean and cov with the help of
        _mean = self.form_posterior_mean(X_star)
        _cov = self.form_posterior_covariance(X_star)

        # draw the samples using normal distribution given posterior mean and covriance
        _draws = np.random.multivariate_normal(mean=_mean,
                                               cov=_cov,
                                               size=n_draws)

        # return mean and drwas with a epsilon
        return _mean, _draws + epsilon

    #

    # --- this is an example kernel: a linear one!
    def kernel(self, X_star, is_train_kernel):
        return self.X.T @ self.X_star

    #


#


# A specific kernel - subclass of Kernel
class SquaredExponentialKernel(Kernel):
    # --- Constructor: length scale should be a d-dimensional vector (one scaling per dimension), and noise variancee is a number
    def __init__(self, X, y, length_scale, noise_variance):
        # initiate a base kernel for computing the cholesky decomp and kernel matrix
        super(SquaredExponentialKernel, self).__init__(X, y)

        # init hyper params
        self.length_scale, self.noise_variance = length_scale, noise_variance

        # init kernel in training mode
        self.K = self.kernel(self.X, self.X, True)

    #

    # --- compute squared exponential kernel between two data matrices: X_1 and X_2
    #   (1) is_train_kernel is a boolean flag indicating whether we are computing the kernel on _just_ the training data,
    #       in which case we add the noise variance to the diagonal
    def kernel(self, X_1, X_2, is_train_kernel=False):
        # To compute the square kernel we will use vector form and compute the eucledian distance using following formulae
        # (a-b)^2 = a^2 + b^2 - 2ab
        eculedian_distance = np.sum(X_1**2, axis=1).reshape(-1, 1) + np.sum(
            X_2**2, axis=1) - 2 * X_1 @ X_2.T

        # compute the kernel function
        _kernel = (np.exp(-0.5 * (1 / self.length_scale) * eculedian_distance))

        # return with noise term if in training mode
        if is_train_kernel:
            # compute noise term sigma^2I
            noise_term = self.noise_variance**2 * (np.eye(N=_kernel.shape[0],
                                                          M=_kernel.shape[1]))

            # add noise_term in the kernel in training mode
            _kernel += noise_term

        return _kernel

    #


#
