"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Exposes traditional ML scikit-learn models.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_validate, KFold, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

from microML.settings import FIGURE_DIR, N_JOBS
from microML.model.ml import ML


logger = logging.getLogger("root_logger")
matplotlib.use('Agg')
matplotlib.rc('font', family="Helvetica")
matplotlib.rcParams["agg.path.chunksize"] = 10000


class Sklearn(ML):

    def __init__(self,
                 features: list[str],
                 target: str,
                 ) -> None:
        """

        Parameters
        -------------
        features: list[str]
            List of model features (predictors) name.
        target: str
            Target variable name (response function).
        """
        super().__init__(features=features, target=target)
        return

    def score_model(self,
                    model,
                    X: np.ndarray,
                    y: np.ndarray,
                    scoringMetric: str = "neg_mean_squared_error",
                    n_folds: int = 10
                    ):
        """
        Runs cross-validation on the model and training dataset.
        """
        logger.info(f"Cross validation dataset size: {str(X.shape)}")
        logger.info("Model validation ... ")
        cvs = cross_validate(model,
                             X,
                             y,
                             cv=KFold(n_folds, shuffle=True),
                             scoring=scoringMetric,
                             return_estimator=True,
                             n_jobs=N_JOBS
                             )
        scores, estimators = cvs["test_score"], cvs["estimator"]
        fit_times, score_times = cvs["fit_time"], cvs["score_time"]
        logger.info("Score (%s): %4.4f +/- %4.4f" % (scoringMetric, scores.mean(), scores.std()))
        return scores, estimators, fit_times, score_times

    def linear_expression(self, coefs_mean, coefs_std, intercept_mean, intercept_std):
        """
        Express the trained linear model in form an equation.
        """
        if len(self.features) != len(coefs_mean):
            msg = "Number of features does not match with the number of fit coefficients."
            logger.error(msg)
            raise ValueError(msg)

        sortIndeces = np.argsort(-np.abs(coefs_mean))
        coefs_mean = [coefs_mean[i] for i in sortIndeces]
        coefs_std = [coefs_std[i] for i in sortIndeces]
        features = [self.features[i] for i in sortIndeces]

        expression = self.target + " = "
        std = "std terms = "
        for i in range(len(features)):
            expression += " %+ 2.2f %s" % (coefs_mean[i], features[i])
            std += " %+ 2.2f%s" % (coefs_std[i], features[i])
        expression += " %+ 2.2f" % intercept_mean
        std += " %+2.2f" % intercept_std
        return expression, std

    def linear_regression(self, **kwargs):
        """
        Construct a linear regression model.
        """
        logger.info(f"Linear model for {self.target}")
        self.model = LinearRegression(fit_intercept=True, n_jobs=N_JOBS)
        scores, estimators, fit_times, score_times = self.score_model(self.model,
                                                                      self.X,
                                                                      self.y,
                                                                      scoringMetric="neg_mean_squared_error",
                                                                      n_folds=10)
        self.model.fit(self.X, self.y)
        logger.info("Model coefficients:")
        for ind, estimator in enumerate(estimators):
            if ind == 0:
                coefs, inters = estimator.coef_, estimator.intercept_
            else:
                coefs = np.vstack((coefs, estimator.coef_))
                inters = np.vstack((inters, estimator.intercept_))
        coefs_mean = np.mean(coefs, axis=0)
        coefs_std = np.std(coefs, axis=0)
        intercept_mean = np.mean(inters, axis=0)
        intercept_std = np.std(inters, axis=0)
        expression, expression_std = self.linear_expression(coefs_mean,
                                                            coefs_std,
                                                            intercept_mean,
                                                            intercept_std)
        print("*****************************")
        print(expression)
        print(expression_std)
        print("*****************************")
        return self.model

    def ensemble(self, model_name, plot_importance=True):
        """
        Construct an ensemble (random forest (rf), extra trees regressor(extra), gradient boost (gb)) model.
        """
        print(f"******* {model_name} Model for {self.target} *******")
        if model_name.lower() == "rf":
            self.model = RandomForestRegressor(200, max_features="sqrt", n_jobs=N_JOBS, min_samples_split=20)
        elif model_name.lower() == "extra":
            self.model = ExtraTreesRegressor(200, max_features="sqrt", n_jobs=N_JOBS, min_samples_split=20)
        elif model_name.lower() == "gb":
            self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, loss="ls")
        scores, estimators, fit_times, score_times = self.score_model(self.model,
                                                                      self.X,
                                                                      self.y,
                                                                      scoringMetric="neg_mean_squared_error",
                                                                      n_folds=10)
        self.model.fit(self.X, self.y)
        if plot_importance:
            print('Plotting Feature Importances ... ')
            sorted_features, sorted_scores = self.RF_feature_importance_MDA(self.model, self.features, self.X, self.y)
            self.plot_importance(sorted_features, sorted_scores, self.target, method='MDA')
            sorted_features, sorted_scores = self.RF_feature_importance_MDI(self.model, self.features, self.X, self.y)
            self.plot_importance(sorted_features, sorted_scores, self.target, method='MDI')
        return self.model, None

    @staticmethod
    def RF_feature_importance_MDA(rf, features, X, Y):
        """
        Compute feature importance based on mean decrease accuracy.
        """
        scores = defaultdict(list)
        rs = ShuffleSplit(n_splits=10, test_size=.3)
        for train_idx, test_idx in rs.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            r = rf.fit(X_train, Y_train)
            acc = r2_score(Y_test, rf.predict(X_test))
            for i in range(X.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, rf.predict(X_t))
                scores[features[i]].append((acc-shuff_acc)/acc)
        zipped = sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)
        return list(list(zip(*zipped))[1]), list(list(zip(*zipped))[0])

    @staticmethod
    def RF_feature_importance_MDI(rf, features, X, Y):
        """
        Compute feature importance based on mean decrease impurity.
        """        
        rf.fit(X, Y)
        zipped = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), features), reverse=True)
        return list(list(zip(*zipped))[1]), list(list(zip(*zipped))[0])

    @staticmethod
    def plot_importance(sorted_features, sorted_scores, target, method):
        """
        Plot feature importances (applicable to decision tree based ensemble models).
        """
        os.makedirs(FIGURE_DIR, exist_ok=True)
        plt.figure(figsize=(15, 5))
        plt.clf()
        plt.plot(sorted_features, sorted_scores, 'o')
        plt.xticks(sorted_features, sorted_features, rotation=90)
        plt.ylabel('Relative Importance (%s)' % method)
        plt.title(target)
        plt.tight_layout()
        plt.savefig('%simportance_%s_%s.png' % (FIGURE_DIR, target, method))
        # plt.show()
        plt.close()
        return
