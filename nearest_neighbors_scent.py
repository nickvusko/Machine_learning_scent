"""Basic implementation of K-Nearest and Radius neighbor algorithms"""


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline


class KNNGridSearch:
    def __init__(self, x_train, x_test, y_train):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier()

    def run_knn(self):

        operations = [("scaler", self.scaler), ("knn", self.knn)]

        pipe = Pipeline(operations)

        param_grid = {"knn__n_neighbors": list(range(1, 20)), "knn__weights": ["uniform", "distance"]}

        full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')

        full_cv_classifier.fit(self.x_train, self.y_train)

        for key, value in full_cv_classifier.best_estimator_.get_params().items():
            print(f"{key}: {value}")

        return full_cv_classifier.best_estimator_.get_params()


class KNNClassify:
    def __init__(self, x_train, x_test, y_train, n_neighbors, weights):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def run_knn(self):

        operations = [("scaler", self.scaler), ("rnn", self.knn)]

        pipe = Pipeline(operations)

        pipe.fit(self.x_train, self.y_train)

        return pipe.predict(self.x_test)


class RNNGridSearch:
    def __init__(self, x_train, x_test, y_train):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.scaler = StandardScaler()
        self.rnn = RadiusNeighborsClassifier()

    def run_rnn(self):

        operations = [("scaler", self.scaler), ("rnn", self.rnn)]

        pipe = Pipeline(operations)

        param_grid = {"rnn__radius": list(range(1, 20)), "rnn__weights": ["uniform", "distance"]}

        full_cv_classifier = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')

        full_cv_classifier.fit(self.x_train, self.y_train)

        for key, value in full_cv_classifier.best_estimator_.get_params().items():
            print(f"{key}: {value}")

        return full_cv_classifier.best_estimator_.get_params()


class RNNClassify:
    def __init__(self, x_train, x_test, y_train, radius, weights):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.scaler = StandardScaler()
        self.rnn = RadiusNeighborsClassifier(radius=radius, weights=weights)

    def run_rnn(self):

        operations = [("scaler", self.scaler), ("rnn", self.rnn)]

        pipe = Pipeline(operations)

        pipe.fit(self.x_train, self.y_train)

        return pipe.predict(self.x_test)
