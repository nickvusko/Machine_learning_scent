import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from nearest_neighbors_scent import KNNClassify, KNNGridSearch, RNNClassify, RNNGridSearch
from random_forest_scent import RFGridSearch, RFClassify
from pca_scent import PCARun


def plot_matrix(y, y_pred):
    """
    Plot confusion matrix.

    :param y: true class tags
    :param y_pred: predicted class tags
    :return: None
    """
    df_cm = pd.DataFrame(confusion_matrix(y, y_pred), index=[],
                         columns=[])  # edit input names of confusion matrix
    s = sns.heatmap(df_cm, annot=True, cmap="viridis")
    s.set_ylabel("")  # set y label
    s.set_xlabel("")  # set x label
    plt.show()


def show_matrix_plot(x, y):
    """
    Show exploratory matrix.
    On the diagonal, KDE plot are displayed.
    The lower triangle displays scatter plots between variables.

    :param x: variables
    :param y: class tags
    :return: None
    """
    scaler = StandardScaler()
    scaler.fit(x)
    scaled_x = scaler.transform(x)
    df_scaled_x = pd.DataFrame(data=scaled_x, index=x.index, columns=x.columns)
    df_scaled_x["Class"] = y
    g = sns.PairGrid(df_scaled_x, hue="Class", palette="colorblind", corner=True)
    g.map_diag(sns.kdeplot)
    g.map_lower(sns.scatterplot)
    g.add_legend(bbox_to_anchor=(0.4, 0.8), title="")  # edit the title
    g.set(xlim=(-2.5, 1.5), ylim=(-2.5, 1.5))
    plt.show()


if __name__ == "__main__":
    NN = True
    RF = True
    PCA_GO = False
    df = pd.read_csv("data/.txt", sep="\t", header=0, index_col=0)  # edit line - input file

    X = df.drop("Class", axis=1)
    Y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    if NN:
        # show_matrix_plot(X_train, y_train)
        knn_model_best_params = KNNGridSearch(X_train, X_test, y_train).run_knn()
        rnn_model_best_params = RNNGridSearch(X_train, X_test, y_train).run_rnn()

        knn_pred = KNNClassify(X_train, X_test, y_train, knn_model_best_params["knn__n_neighbors"],
                               knn_model_best_params["knn__weights"]).run_knn()
        rnn_pred = RNNClassify(X_train, X_test, y_train, rnn_model_best_params["rnn__radius"],
                               rnn_model_best_params["rnn__weights"]).run_rnn()
        print(f"K-Nearest Neighbour:\n{classification_report(y_test, knn_pred)}")
        plot_matrix(y_test, knn_pred)
        print(f"Radius Nearest Neighbour:\n{classification_report(y_test, rnn_pred)}")
        plot_matrix(y_test, rnn_pred)

    if RF:
        rf_model_best_params = RFGridSearch(X_train, X_test, y_train).run_rf()
        rf_pred = RFClassify(X_train, X_test, y_train, rf_model_best_params["n_estimators"],
                             rf_model_best_params["max_features"], rf_model_best_params["bootstrap"],
                             rf_model_best_params["oob_score"]).run_rf()
        print(f"Random Forest classification:\n{classification_report(y_test, rf_pred)}")
        plot_matrix(y_test, rf_pred)

    if PCA_GO:
        PCA_best_params = PCARun(X, Y, 3).run_pca()
        df_PCA = pd.DataFrame(data=PCA_best_params, index=df.index, columns=[f"component:{x}" for x in range(1, 4)])
        df_PCA.to_csv("data/.txt", sep="\t")  # edit line - name of the output file
