import logging
from typing import Optional
import pandas as pd
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from microML.common import feature_engineering
from microML.settings import TARGETS, N_JOBS


logger = logging.getLogger("root_logger")


class Process():
    def __init__(self,
                 target: str,
                 dataset_path: str,
                 surface_feature_engineer: Optional[bool],
                 column_list: Optional[list[str]] = None,
                 filter: Optional[str] = None,
                 holdout: Optional[tuple[str, str]] = None
                 ) -> None:
        """
        Parameters
        -----------
        target: str
            The name of the target variable.
        dataset_path: str
            Path to the dataset to be used for the ML model.
        surface_feature_engineer: Optional[bool]
            Add engineered features to data. If True, it will add engineered features
            for surface model; if False, will add engineered features for surface+depth
            model. If None, will be ignored and no engineered features are added.
        column_list: Optional[list[str]], default None
            List of column names to include in data. If `None` (default) all
            columns will get included.
        filter: Optional[str], default None
            Any sql-like filter to apply on the input dataset.
            Example: 'depth < 10 and heterotrophic_bacteria_abundance > 0'
        holdout: Optional[tuple[str, str]], default None
            A tuple that consists of the table name (first element) and the cruise name
            to create a holdout dataset (test set). If None, will be ignored.
        """
        self.data = pd.read_csv(dataset_path, low_memory=False)
        self.target = target
        if self.target not in self.data.columns:
            msg = f"The target variable ({self.target}) not found in the dataset."
            logger.exception(msg)
            raise ValueError(msg)
        for c in self.data.columns:
            if c == "time":
                self.data[c] = pd.to_datetime(self.data[c])
            elif c in ["table", "cruise"]:
                self.data[c] = self.data[c].astype("str")
            else:
                self.data[c] = self.data[c].astype("float")
        if filter:
            self.data = self.data.query(filter)

        self.data["month"] = self.data["time"].dt.month
        self.data = self.data.drop("time", axis=1, inplace=False)

        # remove redundant, buggy, unnecessary, or columns with too many missing values
        self.remove_col(cols=["A_mn_clim", "O_mn_clim", "si_mn_clim",
                              "n_mn_clim", "p_mn_clim", "C_mn_clim",
                              "s_mn_clim", "t_mn_clim", "M_mn_clim",
                              "mls_da_argo_clim", "mls_dt_argo_clim",
                              "mlt_da_argo_clim", "mlt_dt_argo_clim",
                              "mlpd_da_argo_clim", "mlpd_dt_argo_clim",                              
                              "mld_da_mean_argo_clim", "mld_dt_mean_argo_clim",
                              "PAR", "Fe", "PP"])

        # only accept positive target values (remove nan and negative)
        self.data = self.data.query(f"{self.target}>=0")
        if surface_feature_engineer is not None:
            self.data = feature_engineering(self.data, surface=surface_feature_engineer)
        if column_list:
            self.data = self.data[column_list + ["table", "cruise"]]

        if holdout:
            holdout_query = f"cruise=='{holdout[1]}'"
            data_query = f"cruise!='{holdout[1]}'"
            if not holdout[1]:
                holdout_query = f"table=='{holdout[0]}'"
                data_query = f"table!='{holdout[0]}'"
            self.holdout_data = self.data.query(holdout_query)
            self.data = self.data.query(data_query)
            self.holdout_data = self.holdout_data.drop(["table", "cruise"], axis=1, inplace=False)
        self.data = self.data.drop(["table", "cruise"], axis=1, inplace=False)
        logger.info(f"Dataset loaded: Row# ({len(self.data)}), Column# ({len(self.data.columns)})")
        return

    def remove_col(self, cols: list[str]):
        """
        Remove columns from data.
        """
        for col in cols:
            if col in self.data.columns:
                self.data = self.data.drop(col, axis=1, inplace=False)
        return

    def remove_other_targets(self):
        for t in TARGETS:
            if t == self.target:
                continue
            self.remove_col(cols=[t])
        return

    def scale(self, include_traget: bool = False):
        """
        standard scaler: transform to mean of zero and standard deviation of 1.
        """
        targetCol = self.data[self.target]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(scaled, index=self.data.index, columns=self.data.columns)
        if not include_traget:
            self.data[self.target] = targetCol
        return scaler

    def nan_percent(self) -> pd.DataFrame:
        """
        Return a dataframe containing the percentage of NAN values in each column of data.
        """
        percent_missing = self.data.isnull().sum() * 100.0 / len(self.data)
        missing_value_df = pd.DataFrame({"column_name": self.data.columns,
                                         "percent_missing": percent_missing})
        missing_value_df.sort_values("percent_missing", ascending=False, inplace=True)        
        return missing_value_df

    def remove_nans(self):
        """
        First drop columns of rows that are completely empty, if any exists.
        Then remove rows with at least one missing value.
        """
        len1 = len(self.data)
        missing_value_df = self.nan_percent()
        logger.info(f"Missing value percentage: {missing_value_df}")
        self.data = self.data.dropna(axis=0, how="all", inplace=False)
        self.data = self.data.dropna(axis=1, how="all", inplace=False)
        self.data = self.data.dropna(axis=0, how="any", inplace=False)
        logger.info(f"NAN removal: dataset length changed from {len1} to {len(self.data)}")
        return

    def remove_outliers(self, target: str, extreme_left: float, extreme_right: float):
        """
        Remove data that fall outside the `extreme` quantile.
        """
        len1 = len(self.data)
        lowerBound = self.data[target].quantile(1-extreme_left)
        higherBound = self.data[target].quantile(extreme_right)
        self.data = self.data.query(f"{target}>{lowerBound} & {target}<{higherBound}")
        logger.info(f"Outlier removal for {target}: dataset length changed from {len1} to {len(self.data)}")
        return

    def run(self, remove_nans: bool, scale_features: bool) -> float:
        """
        Remove abundance columns other that self.target form data (self.data), remove
        outliers from target, and apply Box-Cox transformation on the target column.

        Parameters
        ------------
        remove_nans: bool
            Remove rows with at least one NaN value. Note: If the self.data contains columns representing
            surface values, such as satellite observation, setting this parameter to True will limit the
            data to surface values.
        scale_features: bool
            Apply standard scaler on features.

        Returns
        ----------
        float
            Box-Cox optimized lambda
        """
        self.remove_other_targets()
        if remove_nans:
            self.remove_nans()
        self.remove_outliers(target=self.target, extreme_left=0.99, extreme_right=0.99)

        self.data[self.target], lmbda = boxcox(self.data[self.target], lmbda=None)
        logger.info(f"Box-Cox lambda for {self.target}: {lmbda}")

        if scale_features:
            self.scale(include_traget=False)
        return lmbda

    def feature_selection_rfe(self, n_features: int = 5) -> list[str]:
        """
        Select `n_features` from self.data using the Recursive Feature Elimination (RFE) algorithm.

        Returns
        ---------
        list[str]
            Name of selected features.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop(labels=[self.target], axis=1),
                                                            self.data[self.target],
                                                            test_size=0.2)
        model = RandomForestRegressor(n_estimators=10, n_jobs=N_JOBS)
        selector = RFE(model, n_features_to_select=n_features)
        # selector = SequentialFeatureSelector(model,
        #                                      n_features_to_select=n_features,
        #                                      direction="forward",
        #                                      cv=5,
        #                                      scoring="neg_mean_squared_error",
        #                                      n_jobs=N_JOBS)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[(selector.get_support())]
        return selected_features
