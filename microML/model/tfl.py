"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Exposes traditional ML scikit-learn models.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import r2_score
import tensorflow as tf
from microML.common import pretty_target
from microML.settings import MODEL_TEST_DIR, VERBOSE_FIT, VALIDATION_SPLIT, DEFAULT_NN_HYPER_PARAMS
from microML.model.ml import ML


logger = logging.getLogger("root_logger")
matplotlib.use('Agg')
matplotlib.rc('font', family="Helvetica")
matplotlib.rcParams["agg.path.chunksize"] = 10000


class TensorFlow(ML):

    def __init__(self,
                 features: list[str],
                 target: str,
                 hp: Optional[dict] = None
                 ) -> None:
        """

        Parameters
        -------------
        features: list[str]
            List of model features (predictors) name.
        target: str
            Target variable name (response function).
        hp: optional[dict]
            A dictionary containing the hyper parameters of the model:
            hp = {
                  'features': list[str],
                  'layers': list[dict]
                  'epoch': int,
                  'batch': int,
                  'learning': float,
                  'optimizer': str,
                  'loss': str
                  }
        """
        super().__init__(features=features, target=target)
        self.hp = hp
        if not self.hp:
            self.hp = DEFAULT_NN_HYPER_PARAMS
            self.hp["features"] = self.features
        return

    def fit(self, dir_path=MODEL_TEST_DIR, **kwargs):
        """
        Construct a sequential MLP neural network model.

        Parameters
        -------------
        dir_path: str
            Path to a directory to store the model and visualizations.
        """
        logger.info("******* NN Model for %s *******" % self.target)
        self.model = tf.keras.Sequential()

        layers = self.hp["layers"]
        for i in range(len(layers)):
            if layers[i]["type"] == "dense":
                if i == 0:
                    if layers[i]["activation"] == "selu":
                        self.model.add(tf.keras.layers.Dense(layers[i]["units"],
                                                             kernel_initializer="lecun_normal",
                                                             activation=layers[i]["activation"],
                                                             input_shape=[self.X.shape[1]]
                                                             ))
                    else:
                        self.model.add(tf.keras.layers.Dense(layers[i]["units"],
                                                             activation=layers[i]["activation"],
                                                             input_shape=[self.X.shape[1]]
                                                             ))
                else:
                    if layers[i]["activation"] == "selu":
                        self.model.add(tf.keras.layers.Dense(layers[i]["units"],
                                                             kernel_initializer="lecun_normal",
                                                             activation=layers[i]["activation"]
                                                             ))
                    else:
                        self.model.add(tf.keras.layers.Dense(layers[i]["units"],
                                                             activation=layers[i]["activation"]
                                                             ))

            if layers[i]["type"] == "dropout":
                self.model.add(tf.keras.layers.Dropout(layers[i]["drop"]))

        self.model.add(tf.keras.layers.Dense(1, activation="relu"))

        if self.hp["optimizer"] == "RMSprop":
            opt = tf.keras.optimizers.RMSprop(self.hp["learning"])
        if self.hp["optimizer"] == "Adam":
            opt = tf.keras.optimizers.Adam(self.hp["learning"])

        self.model.compile(loss=self.hp["loss"], optimizer=opt, metrics=["mae", "mse"])
        self.model.summary()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)

        history = self.model.fit(self.X_train,
                                 self.y_train,
                                 validation_split=VALIDATION_SPLIT,
                                 epochs=self.hp["epoch"],
                                 verbose=VERBOSE_FIT,
                                 shuffle=True,
                                 callbacks=[early_stop]
                                 )
        self.loss_curve_plot(history, dir_path=f"{dir_path}")
        self.evaluation_plot(dir_path=f"{dir_path}")
        earlyEpochs = len(pd.DataFrame(history.history))
        return self.model, earlyEpochs

    def loss_curve_plot(self, history: dict, dir_path: str):
        """
        Plots the model's (tensorflow) training history.

        Parameters
        ------------
        history: dict
            loss metrics history stored in a dictionary.
        dir_path: str
            Path to a directory to store the plot.
        """
        os.makedirs(dir_path, exist_ok=True)
        hist = pd.DataFrame(history.history)
        hist["epoch"] = history.epoch
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.plot(hist["epoch"], hist["mse"], "-o", markeredgewidth=0, markersize=6,
                 color="tomato", alpha=0.7, label="Training MSE"
                 )
        plt.plot(hist["epoch"], hist["val_mse"], "-o", markeredgewidth=0, markersize=6,
                 color="dodgerblue", alpha=0.7, label="Validation MSE"
                 )
        plt.title(f"{pretty_target(self.target)} Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend(fontsize=18)
        plt.savefig(f"{dir_path}{self.target}_tf_loss_curve.png", dpi=300)
        plt.close()
        return

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the input x.
        """
        y_pred = self.model.predict(x)
        y_pred = np.squeeze(y_pred)
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
        pred_loss = self.model.evaluate(self.X_test, self.y_test)[0]
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
        plt.savefig(f"{dir_path}{self.target}_tf_eval.png", dpi=300)
        return y_pred, pred_loss



# def save_production_model(target, model, scaler, tFlow, dirPath):
#     """
#     Saves the trained ML models on local disk.

#     `tFlow` is a flag indicating the usage of TensorFlow library. 
#     Set `dirPath` to `None` if you want to save the trained model in the production directory. 
#     """
#     prodDir, modelFname, scalerFname = productionFilenames(target, tFlow, dirPath)
#     makedir(prodDir)
#     joblib.dump(scaler, scalerFname) 


#     if tFlow:
#         model.save(modelFname)
#     else:    
#         joblib.dump(model, modelFname) 
#     return


# def load_production_model(target, tFlow, dirPath):
#     """
#     Loads the trained ML models from local disk.

#     `tFlow` is a flag indicating the usage of TensorFlow library. 
#     Set `dirPath` to `None` if you want to save the trained model in the production directory. 
#     """
#     _, modelFname, scalerFname = productionFilenames(target, tFlow, dirPath)
#     scaler = joblib.load(scalerFname) 

#     if tFlow:
#         model = tf.keras.models.load_model(modelFname)
#     else:    
#         model = joblib.load(modelFname) 
#     return model, scaler

