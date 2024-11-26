from typing import Optional
import logging
from microML.settings import SURFACE_DEPTH, COMPILED_DIR, TARGETS, PROC, SYNC, PICO, HETB
from tqdm.auto import tqdm
from microML.analysis.process import Process
from microML.model.ml import ML
from microML.model.skl import Sklearn
from microML.model.tfl import TensorFlow
from microML.model.pyt import PyTorch
from microML.model.tab import PyTorchTab
from microML.model.xgb import XGB
from microML.analysis.explore import Explore
from microML.common import surface_and_depth_features, surface_features


logger = logging.getLogger("root_logger")


class MicroML():
    def __init__(self,
                 model: ML,
                 features: list[str],
                 target: str,
                 surface: bool,
                 holdout: Optional[list[str, str]] = None,
                 test_size: bool = 0.2,
                 random_state: Optional[int] = None
                 ):
        """
        Process the input dataset and create X_train, y_train, X_test, and y_test for modeling.

        Parameters
        -----------
        model: ML
            An Instance of ML class or its children.
        features: list[str]
            List of model features (predictors) name.
        target: str
            The name of the target variable.
        surface: bool
            If True, a surface model is trained, otherwise all data (surface and depth)
            will be used to train a general model.
        holdout: Optional[tuple[str, str]], default None
            A tuple that consists of the table name (first element) and the cruise name
            to create a holdout dataset (test set). If None, the test dataset will be randomly
            created using the train_test_split.
        test_size: float
            Fraction of data to be used for test set (only applicable to train_test_split approach).
        random_state: Optional[int], default None
            Random seed to split the dataset into train and test sets
            (applicable to train_test_split approach).
        """
        self.model = model
        self.features = features
        self.target = target
        self.surface = surface
        self.holdout = holdout
        self.test_size = test_size
        self.random_state = random_state
        filter = f" depth <= {SURFACE_DEPTH} "if self.surface else None
        process = Process(target=self.target,
                          dataset_path=f"{COMPILED_DIR}compiled.csv",
                          surface_feature_engineer=self.surface,
                          column_list=self.features + [self.target],
                          filter=filter,
                          holdout=self.holdout
                          )
        process.run(remove_nans=True, scale_features=False)
        # create X_train, y_train, X_test, and y_test
        if self.holdout:
            self.model.from_given_testset(train_data=process.data, test_data=process.holdout_data)
        else:
            self.model.from_train_test_split(data=process.data,
                                             test_size=self.test_size,
                                             random_state=self.random_state
                                             )

        return


if __name__ == "__main__":
    surface = False
    features_index = 0
    features = surface_features(index=features_index) if surface else surface_and_depth_features(index=features_index)
    holdout = ("SeaFlow_dataset_v1.6", "KOK1606")
    holdout = None
    for target in TARGETS[:1]:
        # model = TensorFlow(features=features, target=target)
        # model = Sklearn(features=features, target=target)
        # model = PyTorch(features=features, target=target)
        # model = PyTorchTab(features=features, target=target)
        model = XGB(features=features, target=target)
        mml = MicroML(model=model,
                      features=features,
                      target=target,
                      surface=surface,
                      holdout=holdout,
                      test_size=0.2,
                      random_state=None
                      )
        # mml.model.linear_regression()
        # mml.model.ensemble(model_name="extra")
        mml.model.fit()
