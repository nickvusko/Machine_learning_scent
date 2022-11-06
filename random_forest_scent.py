"""Basic implementation of Random Forest algorithm"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class RFGridSearch:
    def __init__(self, x_train, x_test, y_train):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.rf = RandomForestClassifier()

    def run_rf(self):
        param_grid = {"n_estimators": list(range(15, 200, 10)), "max_features": list(range(1, 11, 2)),
                      "bootstrap": [True, False], "oob_score": [True, False]}

        full_cv_classifier = GridSearchCV(self.rf, param_grid, cv=5, scoring='accuracy')

        full_cv_classifier.fit(self.x_train, self.y_train)

        for key, value in full_cv_classifier.best_estimator_.get_params().items():
            print(f"{key}: {value}")

        return full_cv_classifier.best_estimator_.get_params()


class RFClassify:
    def __init__(self, x_train, x_test, y_train, n_estimators, max_features, bootstrap, oob_score):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap,
                                         oob_score=oob_score)

    @property
    def run_rf(self):

        self.rf.fit(self.x_train, self.y_train)

        return self.rf.predict(self.x_test)
