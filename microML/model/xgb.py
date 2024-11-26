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
from matplotlib.ticker import AutoMinorLocator
from collections import defaultdict
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate, KFold, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from microML.common import pretty_target
from microML.settings import FIGURE_DIR, N_JOBS, FONT, MODEL_TEST_DIR
from microML.model.ml import ML


logger = logging.getLogger("root_logger")
matplotlib.use('Agg')
matplotlib.rc('font', family=FONT)
matplotlib.rcParams["agg.path.chunksize"] = 10000


class XGB(ML):

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

    def fit(self, plot_importance=True):
        """
        Construct an xgoost regression model.
        """
        self.model = xgb.XGBRegressor(n_estimators=200,
                                      earl_stopping_rounds=50,
                                      learning_rate=0.001
                                      )

        self.model.fit(self.X_train,
                       self.y_train,
                       eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                       verbose=True
                       )
        
        
        self.evaluation_plot(dir_path=MODEL_TEST_DIR)

        if plot_importance:
            print('Plotting Feature Importances ... ')
            sorted_features, sorted_scores = self.RF_feature_importance_MDA(self.model, self.features, self.X, self.y)
            self.plot_importance(sorted_features, sorted_scores, self.target, method='MDA')
            sorted_features, sorted_scores = self.RF_feature_importance_MDI(self.model, self.features, self.X, self.y)
            self.plot_importance(sorted_features, sorted_scores, self.target, method='MDI')
        return self.model, None


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the input x.
        """
        y_pred = self.model.predict(x)
        return y_pred

    def evaluation_plot(self, dir_path: str) -> float:
        """
        Evaluate the trained model on the test set.
        Create a plot comparing the predictions and actual data.

        Parameters
        ------------
        dir_path: str
            Path to a directory to store the plot.

        Returns
        ---------
        float
            Prediction loss on the test set.
        """
        y_pred = self.predict(self.X_test)
        pred_loss = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        linear_coef = np.polyfit(self.y_test, y_pred, 1)
        linear_func = np.poly1d(linear_coef)
        diag_func = np.poly1d([1, 0])

        os.makedirs(dir_path, exist_ok=True)
        plt.clf()
        plt.figure(figsize=(15, 10))
        plt.plot(self.y_test, y_pred, "o", markeredgewidth=0, markersize=4, color="gray", alpha=0.4, label=f"Test Sample Size: {len(self.y_test)}")
        plt.plot(self.y_test, diag_func(self.y_test), "-", linewidth=2, color="dodgerblue", label="$y=x$")
        plt.plot(self.y_test, linear_func(self.y_test), "-", linewidth=2, color="orangered", label="Linear Fit")
        plt.xlim(np.quantile(self.y_test, 0), np.quantile(self.y_test, 0.99))
        plt.ylim(np.quantile(y_pred, 0), np.quantile(y_pred, 0.99))
        plt.legend(fontsize=18)
        plt.xlabel("Observation", fontsize=20)
        plt.ylabel("Prediction", fontsize=20)
        plt.title(f"{pretty_target(self.target)}\nPrediction loss: {pred_loss:.2f}, $R^2$: {r2:.2f}", fontsize=20)
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().tick_params(axis="both", which="both", direction="in", top=True, right=True, labeltop=False, labelright=False)
        plt.tight_layout()
        plt.savefig(f"{dir_path}{self.target}_xgb_eval.png", dpi=300)
        return y_pred, pred_loss


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
        plt.ylabel("Relative Importance ({method})")
        plt.title(target)
        plt.tight_layout()
        plt.savefig(f"{FIGURE_DIR}xgb_importance_{target}_{method}.png")
        # plt.show()
        plt.close()
        return
