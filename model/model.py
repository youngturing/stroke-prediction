import pickle
import logging

import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
all_models_metrics = []


class LoadAndTransform:
    def __init__(self, path='../data/stroke_data_transformed.csv'):
        self.__path = path

    def _load_data(self) -> pd.DataFrame:
        raw_data = pd.read_csv(self.__path)
        return raw_data

    @staticmethod
    def transform_data(raw_data: pd.DataFrame, use_smote=True):
        """
        Parameters
        ----------
            raw_data: pd.Dataframe with raw from Stroke database
            use_smote: boolean parameter indicating using SMOTE technique
        Returns
        -------
            x_train_smote: training data oversampled with SMOTE algorithm
            y_train_smote: training labels oversampled with SMOTE algorithm
            y_train, y_test: test data and labels without oversampling
        """
        data, labels = raw_data.iloc[:, :5], raw_data['stroke']
        tf.random.set_seed(RANDOM_SEED)
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels,
            test_size=0.3,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        if use_smote:
            oversampled = SMOTE(random_state=RANDOM_SEED)
            x_train_smote, y_train_smote = oversampled.fit_resample(x_train, y_train)
            return x_train_smote, x_test, y_train_smote, y_test
        else:
            return x_train, x_test, y_train, y_test


class Models:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.all_models_metrics = []
        self.__excel_filenames = 'stroke_models_metrics.xlsx'
        self.__metrics_file = './models_metrics.csv'

        self.lda = LinearDiscriminantAnalysis(
            solver='svd', store_covariance=False, tol=0.0001
        )

        self.decision_tree = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            random_state=RANDOM_SEED, splitter='best'
        )

        self.qda = QuadraticDiscriminantAnalysis(
            priors=None, reg_param=0.0,
            store_covariance=False, tol=0.0001
        )

        self.log_reg = LogisticRegression(max_iter=200)
        self.models = [self.lda, self.decision_tree, self.log_reg, self.qda]
        self.model_names = [
            'stroke_model_LDA_scikit.sav', 'stroke_model_DT_scikit.sav',
            'stroke_model_LogReg_scikit.sav', 'stroke_model_QDA_scikit.sav'
        ]

    def train_models_and_save(self) -> None:
        for model, model_name in zip(self.models, self.model_names):
            model = model.fit(self.x_train, self.y_train)
            pickle.dump(model, open(model_name, 'wb'))

    def save_metrics_to_excel(self) -> None:
        writer = pd.ExcelWriter(self.__excel_filenames, engine='openpyxl')
        with writer:
            for model_name, models_metrics in zip(self.model_names, self.all_models_metrics):
                df = pd.DataFrame(models_metrics)
                df.to_excel(writer, sheet_name=f'{model_name}')

    def save_metrics_to_csv(self) -> None:
        metrics_df = pd.DataFrame(data=self.all_models_metrics)
        metrics_df.insert(0, 'Model', self.model_names)
        metrics_df.to_csv(self.__metrics_file, index=False)

    def evaluate(self) -> None:
        for model_name in self.model_names:
            loaded_model = pickle.load(open(model_name, 'rb'))
            y_pred = loaded_model.predict(self.x_test)
            data = {
                'Accuracy': [metrics.accuracy_score(self.y_test, y_pred)],
                'f1_score': [metrics.f1_score(self.y_test, y_pred)],
                'Precision': [metrics.precision_score(self.y_test, y_pred)],
                'Recall': [metrics.recall_score(self.y_test, y_pred)]
            }
            logger.info('Model: {}. Metrics: {}'.format(model_name, data))
            self.all_models_metrics.append(data)
            self.save_metrics_to_excel()
        self.save_metrics_to_csv()


def main():
    load_and_transform = LoadAndTransform()
    raw_data = load_and_transform._load_data()
    x_train, x_test, y_train, y_test = LoadAndTransform.transform_data(raw_data=raw_data, use_smote=True)
    models = Models(x_train, x_test, y_train, y_test)
    models.train_models_and_save()
    models.evaluate()


if '__main__' == __name__:
    main()
