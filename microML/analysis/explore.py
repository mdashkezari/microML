"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Early data exploratory procedures.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from microML.settings import EXPLORE_DIR



class Explore(object):

    def __init__(self, data: pd.DataFrame, target: str) -> None:
        """
        Parameters
        ------------
        data: pandas.DataFrame
            Dataframe containing all columns (features and target).
        target: str
            Target variable name (response function).
        """
        self.data = data
        self.target = target
        self.exp_dir = f"{EXPLORE_DIR}{self.target}/"
        os.makedirs(self.exp_dir, exist_ok=True)
        self.dash_dir = f"{self.exp_dir}dash/"
        os.makedirs(self.dash_dir, exist_ok=True)
        return None

    def describe(self):
        """
        Summary statistics for all columns of the dataframe.
        """
        desc = self.data.describe()
        desc.to_csv(self.exp_dir + "describe.csv")
        return desc

    def box_plot(self):
        """
        Create box plot for all dataframe columns.
        """
        plt.close()
        plt.clf()
        # plt.figure(figsize=(50, 30))
        self.data.boxplot(grid=False)
        plt.savefig(self.exp_dir + "box.png")
        plt.show()
        return

    def pair_plot(self):
        """
        Create a pair-plot for all variables in the dataframe.
        """     
        plt.close()
        plt.clf()
        sns.pairplot(self.data)
        plt.savefig(self.exp_dir + "pair_plot.png")
        plt.show()
        return

    def correlation(self, method="spearman"):
        """"
        Compute pairwise correlation between all columns of dataframe, excluding nan values.
        """
        print("computing pairwise correlations ...")
        self.corr = self.data.corr(method=method)
        return self.corr

    def plot_corr_matrix(self, triangle=True):
        """
        Create a pairwise correlation plot for all variables in the dataframe.
        """
        plt.close()
        plt.clf()
        corr = self.correlation()
        mask = np.zeros((len(corr), len(corr)))
        if triangle:
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, cmap="coolwarm", mask=mask, annot=True, fmt="2.2f", annot_kws={"fontsize": 8})
        plt.savefig(self.exp_dir + "corr_matrix.png")
        plt.show()
        return

    def hists(self):
        """
        Create a histogram plot for all variables in the dataframe.
        """
        for col in self.data.columns:
            if not is_numeric_dtype(self.data[col]):
                continue
            plt.close()
            plt.clf()
            self.data[col].plot.hist(bins=100, alpha=0.8)
            plt.title(col)
            plt.savefig(self.exp_dir + "hist_%s.png" % col)
        return

    def scatters(self):
        """
        Create a scatter plot for the target variable as a function of the rest of
        variables in the dataframe.
        """
        for col in self.data.columns:
            plt.close()
            plt.clf()
            plt.plot(self.data[col], self.data[self.target], "o", ms=1, alpha=0.8)
            plt.xlabel(col)
            plt.ylabel(self.target)
            plt.savefig(self.exp_dir + "scatter_%s.png" % col)
        return

    def plot_spatial_data_points(self):
        """
        A global map showing the spatial distribution of Cyanobacteria observations.
        """
        plt.clf()
        # ax = plt.axes(projection=ccrs.InterruptedGoodeHomolosine())
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=-100))

        plt.plot(
                self.data["lon"],
                self.data["lat"],
                "o",
                markersize=.7,
                markeredgewidth=0,
                color="springgreen",
                alpha=0.7,
                transform=ccrs.PlateCarree()
                )
        gl = ax.gridlines(
                          crs=ccrs.PlateCarree(),
                          draw_labels=False,
                          linewidth=.2,
                          color="k",
                          alpha=0.5,
                          linestyle=":"
                          )
        ax.set_title(f"Sampling distribution {self.target}")
        gl.xlocator = mticker.FixedLocator(np.arange(-180., 240., 60.))
        gl.ylocator = mticker.FixedLocator(np.arange(-90., 120., 30.))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.background_img(name="BM", resolution="fine4")
        # ax.background_img(name="ne_shaded", resolution="low")
        # ax.background_img(name="ne_gray", resolution="low")
        # ax.background_img(name="bath", resolution="low")
        # ax.background_img(name="hyp", resolution="low")
        # ax.stock_img()

        ### draw_labels=False    ## cartopy buggy when central_longitude=-100)
        gl.xlabel_style = {"size": 6, "color": "k"}
        gl.ylabel_style = {"size": 6, "color": "k"}
        plt.savefig(self.exp_dir + "spatial_dist.png", bbox_inches="tight", dpi=600)
        return

    def dashboard(self, fname: Optional[str] = None):
        """
        Create a SweetViz dashboard.

        Parameters
        -----------
        fname: str
            Full path for the generated dashboard file. If `None`, default dir and filename is used.
        """
        rep = sv.analyze(self.data, target_feat=self.target)
        if not fname:
            fname = f"{self.dash_dir}dashboard.html"
        rep.show_html(fname, open_browser=True, layout="vertical")
        return
