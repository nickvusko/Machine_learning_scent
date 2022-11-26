"""Basic implementation of PCA algorithm"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PCARun:
    def __init__(self, var, clas, compos):
        self.var = var
        self.clas = clas
        self.compos = compos
        self.pca_scent = PCA(n_components=self.compos, svd_solver="full")

        self.scaler = StandardScaler()

    def run_pca(self):
        scaled_x = self.scaler.fit_transform(self.var)
        principal_components = self.pca_scent.fit_transform(scaled_x)
        print(self.pca_scent.explained_variance_ratio_)

        plt.figure(figsize=(8, 6))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c=self.clas)
        plt.title("PCA analysis of shot cartridges")
        plt.legend(handles=("Vol1", "Vol2", "Vol3", "Vol4"))
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()
