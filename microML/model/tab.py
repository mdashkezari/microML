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
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, FTTransformerConfig, GANDALFConfig, TabNetModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from microML.common import pretty_target
from microML.settings import MODEL_TEST_DIR, VERBOSE_FIT, VALIDATION_SPLIT, FONT
from microML.model.ml import ML


logger = logging.getLogger("root_logger")
matplotlib.use('Agg')
matplotlib.rc('font', family=FONT)
matplotlib.rcParams["agg.path.chunksize"] = 10000


class PyTorchTab(ML):

    def __init__(self,
                 features: list[str],
                 target: str
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

    def fit(self, dir_path=MODEL_TEST_DIR, **kwargs):
        """
        Construct a sequential MLP neural network model.

        """
        data_config = DataConfig(
            target=[self.target],
            continuous_cols=self.features,
            categorical_cols=[],
        )
        trainer_config = TrainerConfig(
            auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
            batch_size=512,
            max_epochs=100,
        )
        optimizer_config = OptimizerConfig()

        # model_config = CategoryEmbeddingModelConfig(
        #     task="regression",
        #     layers="256-256-512-512",  # 1. current leader
        #     # layers="256-128-128-256-512", # 2. lower loss than 1 but less stable
        #     activation="ReLU",
        #     learning_rate=1e-3,
        # )

        # model_config = FTTransformerConfig(
        #     task="regression",
        #     num_heads=4,
        #     num_attn_blocks=4,
        # )

        # model_config = GANDALFConfig(
        #     task="regression",
        #     gflu_stages=15,
        # )

        model_config = TabNetModelConfig(
            task="regression",
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=3,
        )

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        test_data = self.data.sample(frac=0.2)
        train_data = self.data.drop(test_data.index)
        tabular_model.fit(train=train_data, validation=test_data)
        eval_result = tabular_model.evaluate(test_data)
        pred_df = tabular_model.predict(test_data)

        # tabular_model.save_model("examples/basic")
        # loaded_model = TabularModel.load_model("examples/basic")     

        y_test = test_data[self.target].values    
        y_pred = pred_df[self.target+"_prediction"].values
        pred_loss = eval_result[0]["test_loss"]
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
        plt.savefig(f"{dir_path}{self.target}_tab_eval.png", dpi=300)
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

