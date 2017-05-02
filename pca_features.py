import numpy as np
from sklearn.decomposition import PCA


class PCAAnalysis(object):
    def pca_components(self, data, n_components):
        mean, std = self.moments(data)

        normalized_x = (data - mean) / std  # You need to normalize your data first

        pca_fit = PCA(n_components=n_components).fit(
            normalized_x)  # n_components is the components number after reduction
        print_components = "components:{}, variance{}".format(pca_fit.components_.shape,
                                                              pca_fit.explained_variance_ratio_)
        # print(print_components)
        return pca_fit

    def moments(self, data):
        mean = np.mean(data, 0)
        constant = 1e-10
        std = np.std(data, 0) + constant
        return mean, std

    def transform_inputs(self, components, data):
        return np.dot(data, components.T)
