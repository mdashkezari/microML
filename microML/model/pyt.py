"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Exposes traditional ML scikit-learn models.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
import joblib
from tqdm.auto import tqdm
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import r2_score
import torch
from torch import nn
from microML.common import pretty_target
from microML.settings import MODEL_TEST_DIR, VERBOSE_FIT, VALIDATION_SPLIT, DEFAULT_NN_HYPER_PARAMS, FONT
from microML.model.ml import ML


logger = logging.getLogger("root_logger")
matplotlib.use('Agg')
matplotlib.rc('font', family=FONT)
matplotlib.rcParams["agg.path.chunksize"] = 10000


class TorchModel(nn.Module):
    def __init__(self, in_features: int) -> None:
        """

        Parameters
        ------------
        in_features: int
            Input dimension.
        """
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_features, out_features=256)
        self.dropout1 = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(in_features=256, out_features=256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.layer3 = nn.Linear(in_features=256, out_features=512)
        self.dropout3 = nn.Dropout(p=0.2)
        self.layer4 = nn.Linear(in_features=512, out_features=512)
        self.dropout4 = nn.Dropout(p=0.2)
        self.output_layer = nn.Linear(in_features=512, out_features=1)
        self.activation = nn.ReLU()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.layer1(x))
        x = self.dropout1(x)
        x = self.activation(self.layer2(x))
        x = self.dropout2(x)
        x = self.activation(self.layer3(x))
        x = self.dropout3(x)
        x = self.activation(self.layer4(x))
        x = self.dropout4(x)
        # x = self.activation(self.output_layer(x)) 
        x = self.output_layer(x)
        return x


class PyTorch(ML):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Pytroch model initiated with device: {self.device}")
        self.hp = hp
        if not self.hp:
            self.hp = DEFAULT_NN_HYPER_PARAMS
            self.hp["features"] = self.features
        return

    def get_loss_fn(self):
        return {"mae": nn.L1Loss(),
                "mse": nn.MSELoss(),
                "huber": nn.HuberLoss(),
                }[self.hp["loss"].lower()]

    def fit(self, dir_path=MODEL_TEST_DIR) -> dict:
        self.model = TorchModel(in_features=self.X_train.shape[1]).to(self.device)
        self.loss_fn = self.get_loss_fn()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.hp["learning"])

        self.X_train = np.array(self.X_train, dtype=np.float32)
        self.y_train = np.array(self.y_train, dtype=np.float32)
        self.X_test = np.array(self.X_test, dtype=np.float32)
        self.y_test = np.array(self.y_test, dtype=np.float32)

        self.X_train = torch.from_numpy(self.X_train).to(self.device)
        self.y_train = torch.from_numpy(self.y_train).to(self.device)
        self.X_test = torch.from_numpy(self.X_test).to(self.device)
        self.y_test = torch.from_numpy(self.y_test).to(self.device)
        history = {"epoch": [], "train_loss": [], "test_loss": []}

        for epoch in range(self.hp["epoch"]):
            self.model.train()
            y_pred = self.model(self.X_train)
            loss = self.loss_fn(torch.squeeze(y_pred), self.y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()
            with torch.inference_mode():
                test_pred = self.model(self.X_test)
                test_loss = self.loss_fn(torch.squeeze(test_pred), self.y_test)
            history["epoch"].append(epoch+1)
            history["train_loss"].append(loss)
            history["test_loss"].append(test_loss)
            logger.info(f"Epoch: {epoch+1}/{self.hp['epoch']}, train loss: {loss:.4f}, test loss: {test_loss:.4f}")
        self.loss_curve_plot(history=history, dir_path=dir_path)
        self.evaluation_plot(dir_path=dir_path)
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict using the input x.
        """
        x = x.to(self.device)
        with torch.inference_mode():
            y_pred = self.model(x)
        return y_pred

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
        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.plot(history["epoch"],
                 np.array(torch.tensor(history["train_loss"]).cpu().numpy()),
                 "-o", markeredgewidth=0, markersize=6,
                 color="tomato", alpha=0.7, label="Training Loss"
                 )
        plt.plot(history["epoch"],
                 np.array(torch.tensor(history["test_loss"]).cpu().numpy()),
                 "-o", markeredgewidth=0, markersize=6,
                 color="dodgerblue", alpha=0.7, label="Test Loss"
                 )
        plt.title(f"{pretty_target(self.target)} Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(fontsize=18)
        plt.savefig(f"{dir_path}{self.target}_torch_loss_curve.png", dpi=300)
        plt.close()
        return

    def evaluation_plot(self, dir_path: str) -> float:
        """
        Evaluate the trained model on the test set.
        Create a plot comparing the predictions and actual data.

        Parameters
        ------------
        dir_path: str
            Path to a directory to store the plot.
        """
        y_pred = torch.squeeze(self.predict(self.X_test))
        pred_loss = self.loss_fn(y_pred, self.y_test)
        y_pred = y_pred.cpu().numpy()
        y_test = self.y_test.cpu().numpy()

        # y_test = inv_boxcox(y_test, -0.08521151768826253)
        # y_pred = inv_boxcox(y_pred, -0.08521151768826253)

        # target_transformer = joblib.load(f"{MODEL_TEST_DIR}{self.target}_transformer.joblib")
        # y_test = np.squeeze(target_transformer.inverse_transform(y_test.reshape((len(y_test), 1))))
        # y_pred = np.squeeze(target_transformer.inverse_transform(y_test.reshape((len(y_pred), 1))))


        r2 = r2_score(y_test, y_pred)
        linear_coef = np.polyfit(y_test, y_pred, 1)
        linear_func = np.poly1d(linear_coef)
        diag_func = np.poly1d([1, 0])

        os.makedirs(dir_path, exist_ok=True)
        plt.clf()
        plt.figure(figsize=(15, 10))
        plt.plot(y_test, y_pred, "o", markeredgewidth=0, markersize=4, color="gray", alpha=0.4, label=f"Test Sample Size: {len(y_test)}")
        plt.plot(y_test, diag_func(y_test), "-", linewidth=2, color="dodgerblue", label="$y=x$")
        plt.plot(y_test, linear_func(y_test), "-", linewidth=2, color="orangered", label="Linear Fit")
        plt.xlim(np.quantile(y_test, 0), np.quantile(y_test, 0.99))
        plt.ylim(np.quantile(y_pred, 0), np.quantile(y_pred, 0.99))
        plt.legend(fontsize=18)
        plt.xlabel("Observation", fontsize=20)
        plt.ylabel("Prediction", fontsize=20)
        plt.title(f"{pretty_target(self.target)}\nPrediction loss: {pred_loss:.2f}, $R^2$: {r2:.2f}", fontsize=20)
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().tick_params(axis="both", which="both", direction="in", top=True, right=True, labeltop=False, labelright=False)
        plt.tight_layout()
        plt.savefig(f"{dir_path}{self.target}_torch_eval.png", dpi=300)
        return y_pred, pred_loss
    
    def save_model(self, dir_path: str = MODEL_TEST_DIR):
        """
        Save the trained model and scaler on local disk.

        Parameters
        ------------
        dir_path: str
            Path to the directory where the model is stored.
        """

        os.makedirs(dir_path, exist_ok=True)
        scaler_fname = f"{dir_path}{self.target}_scaler.joblib"
        joblib.dump(self.scaler, scaler_fname) 

        model_fname = f"{dir_path}{self.target}_torch_model.pth"
        torch.save(self.model.state_dict(), model_fname)
        return
