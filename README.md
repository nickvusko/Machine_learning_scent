# Machine_learning_scent
To run a script, run main.py

## Input format

The script takes in .txt files with tabulator as a separator. First column should contain sample names (each row should represent one sample), second should be called 'Class' and contain classification tags.
The rest of the columns should represent variables.

## main.py

3 functions has been implemented so far - Nearest Neighbor (NN) - K- and Radius variation, Random Forest, and Principal Component Analysis.
By default, all 3 are active, to switch some of the component off, go to main.py and:
    change NN = True => to NN = False to switch off NN algorithm
    change RN = True => to RN = False to switch off RN algorithm
    change PCA = True => to PCA = False to switch off PCA algorithm
To select an input for analysis, fill the name of the txt file to line #40: df = pd.read_csv("NAME_OF_TXT", sep="\t", header=0, index_col=0)
For NN and RF, the script trains a model first, and then the model is applied on data. Confusion matrix is displayed as an outcome of the classification.
For PCA, script prints out attributes of the model and shows a score graph of first two principal compounents.

### plot_matrix(y, y_pred)
To properly display a confusion matrix, fill names of the labels  to index and columns argument of line 14 (they should be identical for most of the cases)
  example = df_cm = pd.DataFrame(confusion_matrix(y, y_pred), index=["vol", "vol2", "vol3"], columns=["vol", "vol2", "vol3"]), or
            df_cm = pd.DataFrame(confusion_matrix(y, y_pred), index=["vol", "vol2", "vol3"], columns=index)
            
### show_matrix_plot(x,y)
This is a helper fucntion for quick exploratory analysis. Please note that increasing number of variables increases the size of the matrix plot and computation demands.
If it is desired to skip this function, add # at the beginning of line 48 (comment out)


## nearest_neighbors_scent.py

The script contains two classes for each NN variation: GridSearch for finding the optimal model parameters, and Classify, which applies trained model to classification analysis.
The iput data are normalized.


## random_forest_scent.py

The script contains two classes for RF : RFGridSearch for finding the optimal model parameters, and RFClassify, which applies trained model to classification analysis.
The iput data are normalized.


## pca_scent.py

The script performs PCA analysis.
To edit the legend, fill classes to line 36: ax.legend(handles, ["add", "class", "tags", "here"], title="LEGEND_TITLE")
To edit picture title, edit line 39 (or 40 and 41 to change ax labels)
