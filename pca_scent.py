"""Basic implementation of PCA algorithm"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PCARun:
    def __init__(self, var, clas, compos):
        """
        Create PCA model.

        :param var: variables
        :param clas: class tag
        :param compos: number of principal components
        """
        self.var = var
        self.clas = clas
        self.compos = compos
        self.pca_scent = PCA(n_components=self.compos, svd_solver="full")

        self.scaler = StandardScaler()

    def run_pca(self):
        """
        Run PCA analysis.

        :return: None
        """
        scaled_x = self.scaler.fit_transform(self.var)
        principal_components = self.pca_scent.fit_transform(scaled_x)
        print(f"Explained variance ratios for {self.compos} principal components:\n"
              f"{self.pca_scent.explained_variance_ratio_}")

        print(f"Explained variance for {self.compos} principal components:\n"
              f"{self.pca_scent.explained_variance_}")

        print(f"Singular values for {self.compos} principal components:\n"
              f"{self.pca_scent.singular_values_}")

        print(f"Noise variance for the model:\n"
              f"{self.pca_scent.noise_variance_}")

        fig, ax = plt.subplots()
        scatter = ax.scatter(x=principal_components[:, 0], y=principal_components[:, 1], c=self.clas,
                             cmap="gist_rainbow", edgecolor="black", linewidth=0.5)
        handles, _ = scatter.legend_elements(prop="colors")
        ax.legend(handles, [], title="")  # edit legend text and title
        plt.axhline(y=0, color='k', linewidth=1)
        plt.axvline(x=0, color='k', linewidth=1)
        plt.title("PCA analysis of ...")  # edit the title
        plt.xlabel('First Principal Component')  # edit X-axis label
        plt.ylabel('Second Principal Component')  # edit Y-axis label
        plt.show()
