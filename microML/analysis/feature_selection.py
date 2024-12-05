import logging
from microML.settings import COMPILED_DIR, TARGETS, PROC, SYNC, PICO, HETB
from tqdm.auto import tqdm
from microML.analysis.process import Process
from microML.analysis.explore import Explore
from microML.common import surface_and_depth_features, surface_features


logger = logging.getLogger("root_logger")


def first_cut_feature_selection():
    """
    The first round of feature selection, applied on the full pool
    of features.
    """
    feature_dict = dict({})
    for target in tqdm(TARGETS, leave=False):
        pp = Process(target=target,
                     dataset_path=f"{COMPILED_DIR}compiled.csv",
                     surface_feature_engineer=None,
                     column_list=None,
                     filter=None
                     )
        pp.run(remove_nans=False, scale_features=False)
        # ex = Explore(data=pp.data, target=target)
        # ex.plot_spatial_data_points()
        # ex.dashboard(fname=None)
        features = pp.feature_selection_rfe(n_features=8)
        feature_dict[target] = features
        logger.info(f"{target} features: {features}")
    return feature_dict


def second_cut_feature_selection(surface: bool):
    """
    The second round of feature selection, applied on the first pool
    of selected features plus the engineered features.
    """
    feature_dict = dict({})
    features_round_index = 0
    for target in tqdm(TARGETS, leave=False):
        if surface:
            # first round (index=0) selected features + target
            column_list = surface_features(index=features_round_index) + [target]
            remove_nans = True
        else:
            # first round (index=0) selected features + target
            column_list = surface_and_depth_features(index=features_round_index) + [target]
            remove_nans = False
        pp = Process(target=target,
                     dataset_path=f"{COMPILED_DIR}compiled.csv",
                     surface_feature_engineer=surface,
                     column_list=column_list,
                     filter=None
                     )
        pp.run(remove_nans=remove_nans, scale_features=False)
        features = pp.feature_selection_rfe(n_features=8)
        feature_dict[target] = features
        logger.info(f"{target} features: {features}")
    return feature_dict




logger.info(f"first cut features: {first_cut_feature_selection()}")


# print("\n\n---------------------- Surface Features ----------------------\n\n")
# surface_feature_dict = second_cut_feature_selection(surface=True)
# print("\n\n---------------------- Surface+Depth Features ----------------------\n\n")
# total_feature_dict = second_cut_feature_selection(surface=False)


# print(f"\n\n{'*'*50}  Results  {'*'*50}\n\n")
# print("\nSurface features:")
# print(surface_feature_dict)
# print("\nSurface & Depth features:")
# print(total_feature_dict)

