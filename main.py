import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from RNN_scent import RNNClassify, RNNGridSearch
from KNN_scent import KNNClassify, KNNGridSearch


def plot_matrix(y, y_pred):
    df_cm = pd.DataFrame(confusion_matrix(y, y_pred), index=["czech", "ind", "viet"],
                         columns=["czech", "ind", "viet"])
    s = sns.heatmap(df_cm, annot=True, cmap="viridis")
    s.set_ylabel("Etnicita")
    s.set_xlabel("Přiřazená etnicita")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/ratio_area_30.txt", sep="\t", header=0, index_col=0)

    X = df.drop("ethnicity", axis=1)
    Y = df["ethnicity"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    knn_model_best_params = KNNGridSearch(X_train, X_test, y_train, y_test).run_knn()
    rnn_model_best_params = RNNGridSearch(X_train, X_test, y_train, y_test).run_rnn()

    knn_pred = KNNClassify(X_train, X_test, y_train, y_test, knn_model_best_params["knn__n_neighbors"],
                           knn_model_best_params["knn__weights"]).run_knn()
    rnn_pred = RNNClassify(X_train, X_test, y_train, y_test, rnn_model_best_params["rnn__radius"],
                           rnn_model_best_params["rnn__weights"]).run_rnn()
    plot_matrix(y_test, knn_pred)
    plot_matrix(y_test, rnn_pred)
