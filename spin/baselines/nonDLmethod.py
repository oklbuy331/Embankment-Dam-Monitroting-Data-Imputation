# -*- coding: utf-8 -*-
from fancyimpute import KNN, MatrixFactorization
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split
from tsl.utils import  numpy_metrics


class MiceImputer(object):

    def __init__(self, seed_values=True, seed_strategy="mean", copy=True):
        self.strategy = seed_strategy  # seed_strategy in ['mean','median','most_frequent', 'constant']
        self.seed_values = seed_values  # seed_values = False initializes missing_values using not_null columns
        self.copy = copy
        self.imp = SimpleImputer(strategy=self.strategy, copy=self.copy)

    def fit_transform(self, X, method='Linear', iter=5, verbose=True):

        # Why use Pandas?
        # http://gouthamanbalaraman.com/blog/numpy-vs-pandas-comparison.html
        # Pandas < Numpy if X.shape[0] < 50K
        # Pandas > Numpy if X.shape[0] > 500K

        # Data necessary for masking missing-values after imputation
        null_cols = X.columns[X.isna().any()].tolist()
        null_X = X.isna()[null_cols]

        ### Initialize missing_values

        if self.seed_values:

            # Impute all missing values using SimpleImputer
            if verbose:
                print('Initilization of missing-values using SimpleImputer')
            new_X = pd.DataFrame(self.imp.fit_transform(X))
            new_X.columns = X.columns
            new_X.index = X.index

        else:

            # Initialize a copy based on value of self.copy
            if self.copy:
                new_X = X.copy()
            else:
                new_X = X

            not_null_cols = X.columns[X.notna().any()].tolist()

            if verbose:
                print('Initilization of missing-values using regression on non-null columns')

            for column in null_cols:

                null_rows = null_X[column]
                train_x = new_X.loc[~null_rows, not_null_cols]
                test_x = new_X.loc[null_rows, not_null_cols]
                train_y = new_X.loc[~null_rows, column]

                if X[column].nunique() > 2:
                    m = LinearRegression(n_jobs=-1)
                    m.fit(train_x, train_y)
                    new_X.loc[null_rows, column] = pd.Series(m.predict(test_x))
                    not_null_cols.append(column)

                elif X[column].nunique() == 2:
                    m = LogisticRegression(n_jobs=-1, solver='lbfgs')
                    m.fit(train_x, train_y)
                    new_X.loc[null_rows, column] = pd.Series(m.predict(test_x))
                    not_null_cols.append(column)

        ### Begin iterations of MICE

        model_score = {}

        for i in range(iter):
            if verbose:
                print('Beginning iteration ' + str(i) + ':')

            model_score[i] = []

            for column in null_cols:

                null_rows = null_X[column]
                not_null_y = new_X.loc[~null_rows, column]
                not_null_X = new_X[~null_rows].drop(column, axis=1)

                train_x, val_x, train_y, val_y = train_test_split(not_null_X, not_null_y, test_size=0.2,
                                                                  random_state=42)
                test_x = new_X.drop(column, axis=1)

                if new_X[column].nunique() > 2:
                    if method == 'Linear':
                        m = LinearRegression(n_jobs=-1)
                    elif method == 'Ridge':
                        m = Ridge()

                    m.fit(train_x, train_y)
                    model_score[i].append(m.score(val_x, val_y))
                    new_X.loc[null_rows, column] = pd.Series(m.predict(test_x), index=new_X.index).loc[null_rows].values
                    # new_X.loc[null_rows, column] = pd.Series(m.predict(test_x))
                    if verbose:
                        print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))

                elif new_X[column].nunique() == 2:
                    if method == 'Linear':
                        m = LogisticRegression(n_jobs=-1, solver='lbfgs')
                    elif method == 'Ridge':
                        m = RidgeClassifier()

                    m.fit(train_x, train_y)
                    model_score[i].append(m.score(val_x, val_y))
                    new_X.loc[null_rows, column] = pd.Series(m.predict(test_x))
                    if verbose:
                        print('Model score for ' + str(column) + ': ' + str(m.score(val_x, val_y)))

            if model_score[i] == []:
                model_score[i] = 0
            else:
                model_score[i] = sum(model_score[i]) / len(model_score[i])

        return new_X


if __name__ == '__main__':

    filename = '/home/oklbuy/PycharmProjects/spin/spin/datasets/data.csv'
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    training_mask = np.load('/home/oklbuy/PycharmProjects/spin/spin/datasets/training_mask.npy').astype('bool')
    training_df = df.where(training_mask, np.nan)
    eval_mask = np.load('/home/oklbuy/PycharmProjects/spin/spin/datasets/eval_mask.npy').astype('bool')

    # sensor = np.array(['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12',
    #                    'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20'])
    #
    # coordinates = np.array([[-12, 1280, 0], [-3.5, 1280, 0], [8, 1280, 0],
    #                         [-12, 1290, 0], [-3.5, 1290, 0], [8.5, 1290, 0],
    #                         [-12, 1305, 0], [-3.5, 1305, 0], [6.5, 1305, 0],
    #                         [-12, 1280, 55], [-3.5, 1280, 55], [8, 1280, 55],
    #                         [-12, 1290, 55], [-3.5, 1290, 55], [8.5, 1290, 55],
    #                         [-12, 1305, 55], [-3.5, 1305, 55], [6.5, 1305, 55],
    #                         [-2.5, 1265, 0], [2.5, 1265, 0]], dtype='float32')

    imputed_mean = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(training_df), index=df.index, columns=df.columns)
    mean_mae = []; mean_mse = []; mean_mre = []
    mean_eval_MAE = numpy_metrics.masked_mae(imputed_mean.values, df.values, eval_mask)
    for i in range(imputed_mean.values.shape[1]):
        mean_mae = np.append(mean_mae, numpy_metrics.masked_mae(imputed_mean.values[:, i], df.values[:, i], eval_mask[:, i]))
    mean_eval_MSE = numpy_metrics.masked_mse(imputed_mean.values, df.values, eval_mask)
    for i in range(imputed_mean.values.shape[1]):
        mean_mse = np.append(mean_mse, numpy_metrics.masked_mse(imputed_mean.values[:, i], df.values[:, i], eval_mask[:, i]))
    mean_eval_MRE = numpy_metrics.masked_mre(imputed_mean.values, df.values, eval_mask)
    for i in range(imputed_mean.values.shape[1]):
        mean_mre = np.append(mean_mre, numpy_metrics.masked_mre(imputed_mean.values[:, i], df.values[:, i], eval_mask[:, i]))
    print('Method: Mean MAE: %.2f MSE: %.2f MRE: %.2f' % (mean_eval_MAE, mean_eval_MSE, mean_eval_MRE))

    imputed_mice = MiceImputer().fit_transform(training_df, verbose=False)
    mice_mae = []; mice_mse = []; mice_mre = []
    mice_eval_MAE = numpy_metrics.masked_mae(imputed_mice.values, df.values, eval_mask)
    for i in range(imputed_mice.values.shape[1]):
        mice_mae = np.append(mice_mae, numpy_metrics.masked_mae(imputed_mice.values[:, i], df.values[:, i], eval_mask[:, i]))
    mice_eval_MSE = numpy_metrics.masked_mse(imputed_mice.values, df.values, eval_mask)
    for i in range(imputed_mice.values.shape[1]):
        mice_mse = np.append(mice_mse, numpy_metrics.masked_mse(imputed_mice.values[:, i], df.values[:, i], eval_mask[:, i]))
    mice_eval_MRE = numpy_metrics.masked_mre(imputed_mice.values, df.values, eval_mask)
    for i in range(imputed_mice.values.shape[1]):
        mice_mre = np.append(mice_mre, numpy_metrics.masked_mre(imputed_mice.values[:, i], df.values[:, i], eval_mask[:, i]))
    print('Method: MICE MAE: %.2f MSE: %.2f MRE: %.2f' % (mice_eval_MAE, mice_eval_MSE, mice_eval_MRE))

    # imputed_knn = pd.DataFrame(KNN(k=5, verbose=True).fit_transform(df), index=df.index, columns=df.columns)
    # knn_eval_MAE = numpy_metrics.masked_mae(imputed_knn, df.values, eval_mask)
    # knn_eval_MSE = numpy_metrics.masked_mse(imputed_knn.values, df.values, eval_mask)
    # knn_eval_MRE = numpy_metrics.masked_mre(imputed_knn.values, df.values, eval_mask)
    # print('Method: KNN MAE: %.2f MSE: %.2f MRE: %.2f' % (knn_eval_MAE, knn_eval_MSE, knn_eval_MRE))

    imputed_mf = MatrixFactorization(max_iters=200, shrinkage_value=0.05, verbose=False).solve(training_df,
                                                                                               ~training_mask)
    imputed_mf = pd.DataFrame(imputed_mf, index=df.index, columns=df.columns)
    mf_mae = []; mf_mse = []; mf_mre = []
    mf_eval_MAE = numpy_metrics.masked_mae(imputed_mf.values, df.values, eval_mask)
    for i in range(imputed_mf.values.shape[1]):
        mf_mae = np.append(mf_mae, numpy_metrics.masked_mae(imputed_mf.values[:, i], df.values[:, i], eval_mask[:, i]))
    mf_eval_MSE = numpy_metrics.masked_mse(imputed_mf.values, df.values, eval_mask)
    for i in range(imputed_mf.values.shape[1]):
        mf_mse = np.append(mf_mse, numpy_metrics.masked_mse(imputed_mf.values[:, i], df.values[:, i], eval_mask[:, i]))
    mf_eval_MRE = numpy_metrics.masked_mre(imputed_mf.values, df.values, eval_mask)
    for i in range(imputed_mf.values.shape[1]):
        mf_mre = np.append(mf_mre, numpy_metrics.masked_mre(imputed_mf.values[:, i], df.values[:, i], eval_mask[:, i]))
    print('Method: MF MAE: %.2f MSE: %.2f MRE: %.2f' % (mf_eval_MAE, mf_eval_MSE, mf_eval_MRE))
    pass
