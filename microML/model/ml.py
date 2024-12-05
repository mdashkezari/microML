"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Base ML class to provide train and test datasets.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from microML.settings import PRODUCTION_DIR

logger = logging.getLogger("root_logger")


class ML(object):

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
        self.features = features
        self.target = target
        self.model = None
        return

    def from_train_test_split(self,
                              data: pd.DataFrame,
                              test_size: float,
                              random_state: Optional[int] = None
                              ):
        """
        Construct the train and test datasets via train_test_split function.

        Parameters
        -------------
        data: pandas.DataFrame
            Dataframe containing all train and test samples (with features and target).
        test_size: float
            Fraction of data to be used for test set.
        random_state: Optional[int]
            Random seed to split the dataset into train and test sets.
        """
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.X = np.array(data.drop(labels=[self.target], axis=1, inplace=False))
        self.y = np.array(data[self.target])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state
                                                                                )
        self.scale()
        return

    def from_given_testset(self,
                           train_data: pd.DataFrame,
                           test_data: pd.DataFrame
                           ):
        """
        Construct the train and test datasets using a given test dataset.

        Parameters
        -------------
        train_data: pandas.DataFrame
            Dataframe containing all training data (features and target).
        test_data: pandas.DataFrame
            Dataframe containing all test data (features and target).
        """
        self.data = pd.concat([train_data, test_data], axis=0)

        self.X_train = np.array(train_data[self.features])
        self.y_train = np.array(train_data[self.target])
        self.X_test = np.array(test_data[self.features])
        self.y_test = np.array(test_data[self.target])

        self.X = np.concatenate((self.X_train, self.X_test), axis=0)
        self.y = np.concatenate((self.y_train, self.y_test), axis=0)
        self.scale()
        return

    def scale(self):
        """
        Apply standard scaler on train and test datasets (transform to
        mean of zero and standard deviation of 1).
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.X = self.scaler.transform(self.X)
        return

    def save(self):
        os.makedirs(f"{PRODUCTION_DIR}", exist_ok=True)
        joblib.dump(self.scaler, f"{PRODUCTION_DIR}{self.target}_scaler.joblib") 
        if isinstance(self.model, sklearn.ensemble._forest.ExtraTreesRegressor):
            joblib.dump(self.model, f"{PRODUCTION_DIR}{self.target}_extraTreeRegressor.joblib") 
        return
