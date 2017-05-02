import numpy as np
from sklearn.decomposition import PCA


class PCAAnalysis(object):
    def pca_components(self, data, n_components):
        mean, std = self.moments(data)

        normalized_x = (data - mean) / std  # You need to normalize your data first
        print(normalized_x.shape)

        pca_fit = PCA(n_components=n_components).fit(
            normalized_x)  # n_components is the components number after reduction
        print_components = "components:{}".format(pca_fit.components_.shape)
        print(print_components)
        print('pca_fit:{}'.format(pca_fit))
        return pca_fit

    def moments(self, data):
        mean = np.mean(data, 0)
        constant = 1e-10
        std = np.std(data, 0) + constant
        print_moments = "mean {}, std:{}".format(mean.shape, std.shape)
        print(print_moments)
        return mean, std

    def transform_inputs(self, components, data):
        return np.dot(data, components.T)
