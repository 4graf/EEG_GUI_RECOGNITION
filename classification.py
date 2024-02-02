import pickle

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


class EEGClassifier:
    """Class to make several predictions based on eeg data,
    as well as pre-process the data and reduce its dimension"""

    def __init__(self, model_dir: str):
        self.knn_loaded_model = pickle.load(open(f'{model_dir}/knn', 'rb'))
        self.dtc_loaded_model = pickle.load(open(f'{model_dir}/dtc', 'rb'))
        self.rfc_loaded_model = pickle.load(open(f'{model_dir}/rfc', 'rb'))

    @staticmethod
    def load_data(df: pd.DataFrame = None, file_name: str = None):
        """Loads EEG data from file and divides it into features and targets"""
        if df is not None:
           pass
        elif file_name:
            df = pd.read_excel(file_name)
        else:
            raise Exception('Параметр df или file_name обязаны быть не None.')

        X = df.iloc[:, 5:]
        y = df.iloc[:, 1:4]
        return X, y

    @staticmethod
    def preprocess_data(X, n_components=20):
        """Accepts as input the data matrix X and the optional parameter
        n_components, indicating the number of main components to be left after
        preprocessing.
        As a result, the method returns an X_train_pca matrix containing
        a reduced number of main components according to the specified
        value of n_components."""

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=20)
        X_train_pca = pca.fit_transform(X_scaled)
        return X_train_pca

    @staticmethod
    def calc_metrics(y_real, y_pred):
        """Calculates metrics: true negative, false positive,
        false negative, true positive, accuracy, precision,
        recall and f1"""

        tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
        acc = accuracy_score(y_real, y_pred)
        prec = precision_score(y_real, y_pred)
        rec = recall_score(y_real, y_pred)
        f1 = f1_score(y_real, y_pred)
        return [tn, fp, fn, tp], acc, prec, rec, f1

    def predict_eeg(self, X, task: str):
        """Accepts the matrix X and the string task as parameters and
        performs a prediction based on the task assignment.
        If the task is "real"(if the action is real or imaginary),
        then the KNN model is used.
        If the task is "body_part" (body part definition), then the DecisionTree
        model is used.
        If the task is set to "action" (determining the type of movement),
        then the RandomForest model is used.
        If the task does not match any of these options,
        the method returns -1."""

        if task == 'real':
            pred = self.knn_loaded_model.predict(X)
            return pred
        elif task == 'body_part':
            pred = self.dtc_loaded_model.predict(X)
            return pred
        elif task == 'action':
            pred = self.rfc_loaded_model.predict(X)
            return pred
        else:
            return -1
