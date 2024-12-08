import os
import sys
import glob
import shutil
import logging
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import griddata
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import pycmap
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from microML.settings import PROC, SYNC, PICO, HETB, PRODUCTION_DIR, SPATIAL_PRED_DIR, SPATIAL_PRED_DATA_DIR, SPATIAL_PRED_FIG_DIR, VID_DIR
from microML.common import load_production_model, pretty_target
from microML.tokens import API_KEY

logger = logging.getLogger("root_logger")


def bounds(target):
    if target == PROC:
        return [0, 300000]
    if target == SYNC:
        return [0, 50000]
    if target == PICO:
        return [0, 40000]
    if target == HETB:
        return [0, 3000000]


def cmap(target):
    if target == PROC:
        return cmocean.cm.balance
    if target == SYNC:
        return cmocean.cm.balance
    if target == PICO:
        return cmocean.cm.balance
    if target == HETB:
        return cmocean.cm.balance


def simple_map(df, variable, title, fname=""):
    plt.clf()
    lat = df.lat.unique()
    lon = df.lon.unique()
    shape = (len(lat), len(lon))
    data = df[variable].values.reshape(shape)
    bound = bounds(variable)
    im = plt.imshow(data, extent=[np.min(lon), np.max(lon), np.min(lat), np.max(lat)], cmap="coolwarm", origin="bottom", vmin=bound[0], vmax=bound[1])
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    if len(fname)>0: plt.savefig(fname, dpi=600)
    return


def cartopy_map(df, variable, title, fname=""):
    fig = plt.figure()
    plt.clf()
    lat = df.lat.unique()
    lon = df.lon.unique()
    shape = (len(lat), len(lon))
    data = df[variable].values.reshape(shape)
    bound = bounds(variable)

    ## remove the gap on lon: 180deg
    data[:, 0] = data[:, 2]
    data[:, 1] = data[:, 2]
    lon[-1] = 180
    #################################

    # ax = plt.axes(projection=ccrs.InterruptedGoodeHomolosine())
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=-100))
    
    """
    angle = int(fname.split("/")[-1].split(".png")[0])
    azimuthal, theta = angle, angle
    if angle >= 180: azimuthal = angle - 360
    theta = 90 * np.sin(np.pi*angle/180)

    # ax = plt.axes(projection=ccrs.Orthographic(central_longitude=azimuthal, central_latitude=angle))
    # ax = plt.axes(projection=ccrs.Geostationary(central_longitude=azimuthal, satellite_height=35785831, false_easting=0, false_northing=0))
    ax = plt.axes(projection=ccrs.NearsidePerspective(central_longitude=azimuthal, central_latitude=theta, satellite_height=35785831, false_easting=0, false_northing=0))
    """

    ax.add_feature(cf.COASTLINE)
    ax.add_feature(cf.LAND)
    ax.add_feature(cf.LAKES)


    # im = plt.contourf(lon, lat, data, 60,
    #             transform=ccrs.PlateCarree())


    LON, LAT = np.meshgrid(lon, lat)  

    im = ax.pcolormesh(LON, 
                       LAT, 
                       data, 
                       transform=ccrs.PlateCarree(),
                       vmin=bound[0], 
                       vmax=bound[1], 
                    #    shading="flat", 
                       cmap=cmap(variable), 
                       alpha=1
                       )

    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=.2, 
                      color="k", 
                      alpha=0.5, 
                      linestyle=":"
                      )

    gl.xlocator = mticker.FixedLocator(np.arange(-180.,240.,60.))
    gl.ylocator = mticker.FixedLocator(np.arange(-90.,120.,30.))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ### draw_labels=False    ## buggy when central_longitude=-100)
    gl.xlabel_style = {"size": 6, "color": "k"}
    gl.ylabel_style = {"size": 6, "color": "k"}

    ax.set_title(title)
    ax.background_img(name="BM", resolution="fine%d" % int(title.split("-")[1]) )
    cbar = fig.colorbar(im, fraction=0.04, pad=0.04, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7)
    if len(fname)>0: plt.savefig(fname, dpi=600)
    plt.close()
    return



def video(frameRate, inputPattern, vidName): 
    os.makedirs(VID_DIR, exist_ok=True)
    mov_str =  "ffmpeg " + "  -r " + str(frameRate) + " -i " + inputPattern + " -pix_fmt yuv420p " + VID_DIR + vidName + " -y"
    print(mov_str)
    os.system(mov_str)
    return    


def regrid_map(df, variable, res_x, res_y):
    interpMethod = "linear"
    lat = np.array(df.lat)
    lon = np.array(df.lon)
    depths = df.depth.unique()
    assert len(depths) == 1, f"Only one depth level is allowed in the maps but found {depths}"
    
    points = np.stack((lon, lat), axis=1)
    values = np.array(df[variable])

    lat_min, lat_max = -90, 90
    lon_min, lon_max = -180, 180

    x = np.arange(lon_min, lon_max, res_x)
    y = np.arange(lat_min, lat_max, res_y)
    grid_x, grid_y = np.meshgrid(x, y)
    interpData = griddata(points, values, (grid_x, grid_y), method=interpMethod)

    newDF = pd.DataFrame({})
    newDF["lat"] = grid_y.flatten()
    newDF["lon"] = grid_x.flatten()
    newDF["depth"] = depths[0]
    newDF[variable] = interpData.flatten()
    return newDF 


def get_subset(table, variable, dt1, dt2, lat1, lat2, lon1, lon2, depth1, depth2):
    api = pycmap.API(API_KEY)
    print(f"Downloading {variable} from {table} ...")
    return api.space_time(
                    table=table,
                    variable=variable,
                    dt1=dt1,
                    dt2=dt2,
                    lat1=lat1,
                    lat2=lat2,
                    lon1=lon1,
                    lon2=lon2,
                    depth1=depth1,
                    depth2=depth2
                    )


def global_X(target, dt1, dt2, res_x, res_y):
    lat1, lat2 = -90, 90
    lon1, lon2 = -180, 180
    return global_features_X(dt1, dt2, lat1, lat2, lon1, lon2, res_x, res_y)
    # if target == PROC:
    #     return global_pro_X(dt1, dt2, lat1, lat2, lon1, lon2, res_x, res_y)
    # if target == SYNC:
    #     return global_syn_X(dt1, dt2, lat1, lat2, lon1, lon2, res_x, res_y)
    # if target == PICO:
    #     return global_pico_X(dt1, dt2, lat1, lat2, lon1, lon2, res_x, res_y)




def global_features_X(dt1, dt2, lat1, lat2, lon1, lon2, res_x, res_y):
    pisces_po4 = get_subset("tblPisces_NRT", "PO4", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=0.5)    
    pisces_o2 = get_subset("tblPisces_NRT", "O2", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=0.5)
    pisces_si = get_subset("tblPisces_NRT", "Si", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=0.5)
    pisces_chl = get_subset("tblPisces_NRT", "CHL", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=0.5)

    woa_t = get_subset("tblWOA_Climatology", "sea_water_temp_WOA_clim", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=0)
    woa_con = get_subset("tblWOA_Climatology", "conductivity_WOA_clim", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=0)
    woa_sal = get_subset("tblWOA_2018_qrtdeg_Climatology", "s_an_clim", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=0)

    darwin_dic = get_subset("tblDarwin_Nutrient_Climatology", "DIC_darwin_clim", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=5)
    darwin_nh4 = get_subset("tblDarwin_Nutrient_Climatology", "NH4_darwin_clim", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=5)
    darwin_no2 = get_subset("tblDarwin_Nutrient_Climatology", "NO2_darwin_clim", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=5)
    darwin_no3 = get_subset("tblDarwin_Nutrient_Climatology", "NO3_darwin_clim", dt1, dt2, lat1, lat2, lon1, lon2, depth1=0, depth2=5)

    pisces_po4 = regrid_map(pisces_po4, "PO4", res_x, res_y)
    pisces_o2 = regrid_map(pisces_o2, "O2", res_x, res_y)
    pisces_si = regrid_map(pisces_si, "Si", res_x, res_y)
    pisces_chl = regrid_map(pisces_chl, "CHL", res_x, res_y)

    woa_t = regrid_map(woa_t, "sea_water_temp_WOA_clim", res_x, res_y)
    woa_sal = regrid_map(woa_sal, "s_an_clim", res_x, res_y)
    woa_con = regrid_map(woa_con, "conductivity_WOA_clim", res_x, res_y)

    darwin_dic = regrid_map(darwin_dic, "DIC_darwin_clim", res_x, res_y)
    darwin_nh4 = regrid_map(darwin_nh4, "NH4_darwin_clim", res_x, res_y)
    darwin_no2 = regrid_map(darwin_no2, "NO2_darwin_clim", res_x, res_y)
    darwin_no3 = regrid_map(darwin_no3, "NO3_darwin_clim", res_x, res_y)

    df = pd.DataFrame({})
    df["lat"] = pisces_po4["lat"]
    df["lon"] = pisces_po4["lon"]
    df["depth"] = pisces_po4["depth"]
    df["sea_water_temp_WOA_clim"] = woa_t["sea_water_temp_WOA_clim"]
    df["conductivity_WOA_clim"] = woa_con["conductivity_WOA_clim"]
    df["s_an_clim"] = woa_sal["s_an_clim"]
    df["O2"] = pisces_o2["O2"]
    df["Si"] = pisces_si["Si"]
    df["CHL"] = pisces_chl["CHL"]
    df["PO4"] = pisces_po4["PO4"]
    df["NH4_darwin_clim"] = darwin_nh4["NH4_darwin_clim"]
    df["NO3_darwin_clim"] = darwin_no3["NO3_darwin_clim"]
    df["DIC_darwin_clim"] = darwin_dic["DIC_darwin_clim"]
    df["NO2_darwin_clim"] = darwin_no2["NO2_darwin_clim"]
    return df


def spatialEstimate(df, target, ind):
    lats, lons, depths = df["lat"], df["lon"], df["depth"]
    # df.drop(["lat", "lon"], axis=1, inplace=True)
    df["lon"] = np.cos(df["lon"] * np.pi / 180)
    model, scaler = load_production_model(target)
    # df[target] = 100
    scaled = scaler.transform(df)   
    df = pd.DataFrame(scaled, index=df.index, columns=df.columns)          
    # df.drop([target], axis=1, inplace=True)
    nanPlaceholder = 1e10
    df.fillna(nanPlaceholder, inplace=True)
    XTest = np.array(df)
    y_pred = model.predict(XTest)
    df[target] = y_pred
    for col in df.columns:
        df.loc[df[col]==nanPlaceholder, target] = np.nan
    df["lat"] = lats
    df["lon"] = lons
    df["depth"] = depths
    return df


def Pisces_calendar(api):
    df = api.query("select * from tblPisces_NRT_Calendar")
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].dt.strftime("%Y-%m-%d")
    return df["time"]


def plot_spatial_estimates(dt, target, ind):
    print("*************** " + dt + " *************** \n")
    data_fname = "%s%s/%5.5d.csv" % (SPATIAL_PRED_DATA_DIR, target, ind)
    df = global_X(target, dt, dt, res_x=0.5, res_y=0.5)
    df = spatialEstimate(df, target, ind)
    df.to_csv(data_fname, index=False)

    fig_fname = "%s%s/%5.5d.png" % (SPATIAL_PRED_FIG_DIR, target, ind)
    df = pd.read_csv(data_fname)
    cartopy_map(df, target, title=pretty_target(target)+" [cell/mL]\n"+dt, fname=fig_fname)
    return    



def main(api, targetIndex, vidFlag):
    targets = [PROC, SYNC, PICO, HETB]
    target = targets[targetIndex]
    dts = Pisces_calendar(api)[:6]
    os.makedirs(SPATIAL_PRED_DIR, exist_ok=True)
    os.makedirs(SPATIAL_PRED_DATA_DIR, exist_ok=True)
    os.makedirs(SPATIAL_PRED_FIG_DIR, exist_ok=True)
    for tar in targets:
        os.makedirs(f"{SPATIAL_PRED_DATA_DIR}{tar}/", exist_ok=True)
        os.makedirs(f"{SPATIAL_PRED_FIG_DIR}{tar}/", exist_ok=True)

    print(f"""
           ************************************************************
           *                                                          *  
                Number of {target} maps: {len(dts)} 
           *                                                          *  
           ************************************************************ 
           """)  
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(plot_spatial_estimates, dts, [target] * len(dts), list(range(len(dts))))
    
    ## checks if any frame has been dropped. If so, downloads and creates the frame (map) synchronously. 
    for ind, dt in enumerate(dts):
        framePath = "%s%s/%5.5d.png" % (SPATIAL_PRED_FIG_DIR, target, ind)
        if not os.path.isfile(framePath): plot_spatial_estimates(dt, target, ind)
    
    if vidFlag: video(
                     frameRate=9, 
                     inputPattern=SPATIAL_PRED_FIG_DIR + target + r"/%5d.png", 
                     vidName=target + ".mp4"
                     )
    return








#######################################
#                                     #
#                                     #
#                 main                #
#                                     #
#                                     #
#######################################



if __name__ == "__main__":    
    targetIndex = int(sys.argv[1])
    vidFlag = bool(int(sys.argv[2]))
    api = pycmap.API(API_KEY)
    main(api, targetIndex, vidFlag)