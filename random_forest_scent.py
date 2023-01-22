"""Basic implementation of Random Forest algorithm"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib


class RFGridSearch:
    def __init__(self, x_train, x_test, y_train):
        """
        Perform Grid Search (parameter optimization) for RF.

        :param x_train: training variables
        :param x_test: variables for evaluation
        :param y_train: training class tags
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.rf = RandomForestClassifier()

    def run_rf(self):
        """
        Run Grid Search.

        :return: y_test predictions
        """
        param_grid = {"n_estimators": list(range(15, 200, 10)), "max_features": list(range(1, 11, 2)),
                      "bootstrap": [True, False], "oob_score": [True, False]}

        full_cv_classifier = GridSearchCV(self.rf, param_grid, cv=5, scoring='accuracy')

        full_cv_classifier.fit(self.x_train, self.y_train)

        for key, value in full_cv_classifier.best_estimator_.get_params().items():
            print(f"{key}: {value}")

        return full_cv_classifier.best_estimator_.get_params()


class RFClassify:
    def __init__(self, x_train, x_test, y_train, n_estimators, max_features, bootstrap, oob_score):
        """
        Perform Grid Search (parameter optimization) for RF.

        :param x_train: training variables
        :param x_test: variables for evaluation
        :param y_train: training class tags
        :param n_estimators: number of estimators
        :param max_features: number of max features
        :param bootstrap: [True, False]
        :param oob_score: [True, False]
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap,
                                         oob_score=oob_score)

    def run_rf(self):
        """
        Apply RF model.

        :return: y_test predictions
        """
        self.rf.fit(self.x_train, self.y_train)

        joblib.dump(self.rf, "rf_model.pkl")

        return self.rf.predict(self.x_test)
