"""Basic implementation of PCA algorithm"""
import numpy as np
import pandas as pd
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

    def run_pca(self) -> np.array:
        """
        Run PCA analysis.

        :return: np.array
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

        loadings = self.pca_scent.components_
        n_features = self.pca_scent.n_features_
        pc_loadings = dict(zip([f"PC{i}" for i in list(range(1, n_features+1))], loadings))
        loadings_df = pd.DataFrame.from_dict(pc_loadings)
        loadings_df["features"] = [x for x in pd.read_csv("data/CARTRIDGE.txt", sep="\t", header=0,
                                                          index_col=0).drop("Class", axis=1).columns]
        loadings_df = loadings_df.set_index("features")
        loadings_df.to_csv("data/loadings_PCA_cartridges.txt", sep="\t")

        xs = loadings[0]
        ys = loadings[1]

        # Plot the loadings on a scatterplot
        for i, varnames in enumerate(loadings_df.index):
            plt.scatter(xs[i], ys[i], s=200)
            plt.arrow(
                0, 0,  # coordinates of arrow base
                xs[i],  # length of the arrow along x
                ys[i],  # length of the arrow along y
                color='b',
                head_width=0.01
            )
            plt.text(xs[i], ys[i], varnames)

        # Define the axes
        xticks = np.linspace(-0.8, 0.8, num=5)
        yticks = np.linspace(-0.8, 0.8, num=5)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        # Show plot
        plt.title('2D Loading plot')
        plt.show()
        return principal_components
