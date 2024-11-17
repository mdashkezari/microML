import numpy as np
import pandas as pd
from microML.settings import PROC, SYNC, PICO, HETB, TARGETS


def pretty_target(target: str) -> str:
    """
    Return a more readable target name.

    Returns
    --------
    str
    """
    targets = {PROC: "Prochlorococcus Abundance",
               SYNC: "Synechococcus Abundance",
               PICO: "Picoeukaryotes Abundance",
               HETB: "Heterotrophic Bacteria Abundance"
               }
    return targets.get(target)


def cyano_datasets():
    """
    Compiles a list of Cyanobacteria datasets.
    Each element is a tuple representing a dataset where the first element
    is the table name, the second element is a list of field names to be
    retrieved, the third element is a consisten alias for the measurement
    fields, and the last element is a conversion coefficient ensuring that
    all measurements are in [cell/ml] unit.
    """
    cyanos = []
    # will be replaced manually with v1.6
    # cyanos.append(("tblSeaFlow_v1_5",
    #                ["cruise", "abundance_prochloro", "abundance_synecho", "abundance_picoeuk"],
    #                ["cruise", PROC, SYNC, PICO],
    #                [None, 1e3, 1e3, 1e3],
    #                ))
    cyanos.append(("tblFlombaum",
                  ["prochlorococcus_abundance_flombaum", "synechococcus_abundance_flombaum"],
                  [PROC, SYNC],
                  [1, 1]
                  ))
    cyanos.append(("tblGlobal_PicoPhytoPlankton",
                   ["prochlorococcus_abundance", "synechococcus_abundance", "picoeukaryote_abundance"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblJR19980514_AMT06_Flow_Cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Zubkov", "synechococcus_abundance_P700A90Z_Zubkov", "picoeukaryotic_abundance_PYEUA00A_Zubkov", "bacteria_abundance_HBCCAFTX_Zubkov"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblJR20030512_AMT12_Flow_Cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Zubkov", "synechococcus_abundance_P700A90Z_Zubkov", "picoeukaryotic_abundance_PYEUA00A_Zubkov", "bacteria_abundance_HBCCAFTX_Zubkov"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblJR20030910_AMT13_Flow_Cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Zubkov", "synechococcus_abundance_P700A90Z_Zubkov", "picoeukaryotic_abundance_PYEUA00A_Zubkov", "bacteria_abundance_HBCCAFTX_Zubkov"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblJR20040428_AMT14_Flow_Cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Zubkov", "synechococcus_abundance_P700A90Z_Zubkov", "picoeukaryotic_abundance_PYEUA00A_Zubkov", "bacteria_abundance_TBCCAFTX_Zubkov"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]                  
                   ))
    cyanos.append(("tblD284_AMT15_Flow_Cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Zubkov", "synechococcus_abundance_P700A90Z_Zubkov", "bacteria_abundance_HBCCAFTX_Zubkov"],
                   [PROC, SYNC, HETB],
                   [1, 1, 1]
                   ))  
    cyanos.append(("tblD294_AMT16_Flow_Cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran", "bacteria_abundance_C804B6A6"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblD299_AMT17_Flow_Cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Zubkov", "synechococcus_abundance_P700A90Z_Zubkov", "picoeukaryotic_abundance_PYEUA00A_Zubkov", "bacteria_abundance_TBCCAFTX_Zubkov"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblJR20081003_AMT18_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblJC039_AMT19_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblJC053_AMT20_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblD371_AMT21_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblJC079_AMT22_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblJR20131005_AMT23_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblJR20140922_AMT24_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblJR15001_AMT25_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblDY110_AMT29_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran", "bacteria_abundance_C804B6A6"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))

    cyanos.append(("tblBATS_Bottle", 
                   ["prochlorococcus", "synechococcus", "picoeukaryotes"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblBATS_Bottle_Validation", 
                   ["prochlorococcus", "synechococcus", "picoeukaryotes"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblRR1604_Flow_Cytometry_Mass_Spec", 
                   ["prochlorococcus", "synechococcus", "picoeukaryotes"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))

    cyanos.append(("tblDY084_AMT27_flow_cytometry",
                   ["prochlorococcus_abundance_P701A90Z_Tarran", "synechococcus_abundance_P700A90Z_Tarran", "picoeukaryotic_abundance_PYEUA00A_Tarran", "bacteria_abundance_high_nucleic_acid_cell_content_P18318A9"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblHOT_PP_v2022",
                   ["pbact", "sbact", "ebact", "hbact"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblHOT_Bottle_ALOHA",
                   ["pbact", "sbact", "ebact", "hbact"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))
    cyanos.append(("tblHOT_Bottle_HALE_ALOHA",
                   ["pbact", "sbact", "ebact", "hbact"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))
    cyanos.append(("tblMV1015_cmore_bottle",
                   ["pbact", "sbact", "ebact", "hbact"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))
    cyanos.append(("tblHOT_Flow_Cytometry_Time_Series",
                   ["Pro_abundance", "sbact_cmore", "ebact_cmore", "hbact_cmore"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))
    cyanos.append(("tblEN532_EN538_flow_cytometry",
                   ["Prochlor_cells_per_ml", "Syn_cells_per_ml", "picoEuk_cells_per_ml", "total_het_bact_cells_per_ml"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblLA35A3_flow_cytometry",
                   ["pro", "syn", "peuk", "bacteria"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblKNOX22RR_flow_cytometry",
                   ["Pro", "Syn", "Pico_Euk", "HB"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1]
                   ))
    cyanos.append(("tblMV1008_flow_cytometry_fixed",
                   ["coccus_p", "coccus_s", "peuk"],
                   [PROC, SYNC, PICO],
                   [1, 1, 1]
                   ))
    cyanos.append(("tblKM0704_CMORE_BULA_Underway_Samples",
                   ["prochloro_bact", "synecho_bact", "eukaryotes", "hetero_bact"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))  ## euk (not pico)
    cyanos.append(("tblKM0704_CMORE_BULA_Bottle",
                   ["prochloro_bact", "synecho_bact", "eukaryotes", "hetero_bact"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))  ## euk (not pico)
    cyanos.append(("tblFalkor_2018",
                   ["Prochlorococcus", "Synechococcus", "Eukaryotes", "Heterotrophic_Bacteria"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))  ## euk (not pico)
    cyanos.append(("tblHOT_LAVA",
                   ["Prochlorococcus", "Synechococcus", "Eukaryotes", "Heterotrophic_Bacteria"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))
    cyanos.append(("tblKM1709_mesoscope",
                   ["Prochlorococcus", "Synechococcus", "Eukaryotes", "Heterotrophic_Bacteria"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))  ## euk (not pico)
    cyanos.append(("tblHOE_legacy_2A",
                   ["Prochlorococcus", "Synechococcus", "Eukaryotes", "Heterotrophic_Bacteria"],
                   [PROC, SYNC, PICO, HETB],
                   [1e5, 1e5, 1e5, 1e5]
                   ))  ## euk (not pico)

    # v2024 group
    cyanos.append(("tblInflux_Stations_TN413_2023v1_0",
                   ["cell_abundance_prochloro", "cell_abundance_synecho", "cell_abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblParticle_Abundances",
                   ["pro", "syn", "peuk", "hetbac"],
                   [PROC, SYNC, PICO, HETB],
                   [1e-3, 1e-3, 1e-3, 1e-3],
                   ))
    cyanos.append(("tblInflux_Underway_TN428_2024v1_0",
                   ["cell_abundance_prochloro", "cell_abundance_synecho", "cell_abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblInflux_Stations_Gradients_2023v1_0",
                   ["cell_abundance_prochloro", "cell_abundance_synecho", "cell_abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblInflux_Underway_TN427_2024v1_0",
                   ["cell_abundance_prochloro", "cell_abundance_synecho", "cell_abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblPARAGON1_KM2112_CellAbundances_CTD",
                   ["Pro", "Syn", "Peuk", "Het"],
                   [PROC, SYNC, PICO, HETB],
                   [1, 1, 1, 1],
                   ))
    cyanos.append(("tblInflux_Underway_Gradients_2023v1_0",
                   ["cell_abundance_prochloro", "cell_abundance_synecho", "cell_abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblTN414_Influx_Underway_2023v1_0",
                   ["cell_abundance_prochloro", "cell_abundance_synecho", "cell_abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblTN413_Influx_Underway_2023v1_0",
                   ["cell_abundance_prochloro", "cell_abundance_synecho", "cell_abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblTN397_Gradients4_Influx_Stations_v1_1",
                   ["cell_abundance_prochloro", "cell_abundance_synecho", "cell_abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblTN397_Gradients4_Influx_Underway",
                   ["abundance_prochloro", "abundance_synecho", "abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblKM1906_Gradients_3_Influx_Stations",
                   ["abundance_prochloro", "abundance_synecho", "abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblMGL1704_Gradients_2_Influx_Stations",
                   ["abundance_prochloro", "abundance_synecho", "abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblKOK1606_Gradients_1_Influx_Stations",
                   ["abundance_prochloro", "abundance_synecho", "abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblTN398_Influx_Underway",
                   ["cell_abundance_prochloro", "cell_abundance_synecho", "cell_abundance_picoeuk"],
                   [PROC, SYNC, PICO],
                   [1e3, 1e3, 1e3],
                   ))
    cyanos.append(("tblL4_phytoplankt_nanoflagellates_bacteria_2009",
                   ["synechococcus_spp", "peuk", "hna_het"],
                   [SYNC, PICO, HETB],
                   [1, 1, 1],
                   ))
    cyanos.append(("tblL4_phytoplankt_nanoflagellates_bacteria_2011",
                   ["synechococcus_spp", "peuk", "hna_het"],
                   [SYNC, PICO, HETB],
                   [1, 1, 1],
                   ))
    cyanos.append(("tblL4_phytoplankt_nanoflagellates_bacteria_2010",
                   ["synechococcus_spp", "peuk", "hna_het"],
                   [SYNC, PICO, HETB],
                   [1, 1, 1],
                   ))
    cyanos.append(("tblL4_phytoplankt_nanoflagellates_bacteria_2008",
                   ["synechococcus_spp", "peuk", "hna_het"],
                   [SYNC, PICO, HETB],
                   [1, 1, 1],
                   ))
    cyanos.append(("tblL4_phytoplankt_nanoflagellates_bacteria_2007",
                   ["synechococcus_spp", "peuk", "hna_het"],
                   [SYNC, PICO, HETB],
                   [1, 1, 1],
                   ))
    cyanos.append(("tblSosik_flow_cytometry_NES_LTER",
                   ["syn_cells_per_ml", "redeuk_leq_20um_cells_per_ml"],
                   [SYNC, PICO],
                   [1, 1],
                   ))
    return cyanos


def environmental_datasets():
    """
    Compile a dict of environmental vaiables to be colocalized with species
    measurements. Each item key represents the table name of the environmental
    dataset, and the item's value is itself a dict containing the variables
    names, tolerance parameters, and two flags indicating if the dataset has
    'depth' column, and if the dataset represents a climatology product,
    repectively. The tolerance parametrs specify the temporal [days],
    latitude [deg], longitude [deg], and depth [m] tolerances, respectively.
    """
    envs = {
           "tblSST_AVHRR_OI_NRT": {
                                   "variables": ["sst"],
                                   "tolerances": [0, 0.25, 0.25, 5],
                                   "hasDepth": False,
                                   "isClimatology": False
                                   },
           "tblModis_CHL_cl1": {
                          "variables": ["chlor_a"],
                          "tolerances": [4, 0.25, 0.25, 5],
                          "hasDepth": False,
                          "isClimatology": False
                          },
           "tblModis_POC_cl1": {
                          "variables": ["POC"],
                          "tolerances": [4, 0.25, 0.25, 5],
                          "hasDepth": False,
                          "isClimatology": False
                          },
           "tblSSS_NRT_cl1": {
                          "variables": ["sss_smap"],
                          "tolerances": [0, 0.25, 0.25, 5],
                          "hasDepth": False,
                          "isClimatology": False
                          },
           "tblModis_PAR_cl1": {
                            "variables": ["PAR"],
                            "tolerances": [0, 0.25, 0.25, 5],
                            "hasDepth": False,
                            "isClimatology": False
                            },
           "tblAltimetry_REP_Signal": {
                                "variables": ["sla", "adt", "ugos", "vgos", "ugosa", "vgosa"],
                                "tolerances": [0, 0.25, 0.25, 5],
                                "hasDepth": False,
                                "isClimatology": False
                                },
           "tblPisces_NRT": {
                             "variables": ["NO3", "PO4", "Fe", "O2", "Si", "PP", "CHL", "PHYC"],
                             "tolerances": [4, 0.5, 0.5, 5],
                             "hasDepth": True,
                             "isClimatology": False
                             },
        #    "tblPisces_Forecast_Car": {
        #                      "variables": ["dissic", "ph", "talk"],
        #                      "tolerances": [0, 0.25, 0.25, 5],
        #                      "hasDepth": True,
        #                      "isClimatology": False
        #                      },
        #    "tblPisces_Forecast_Optics": {
        #                      "variables": ["kd"],
        #                      "tolerances": [0, 0.25, 0.25, 5],
        #                      "hasDepth": True,
        #                      "isClimatology": False
        #                      },
        #    "tblPisces_Forecast_Co2": {
        #                      "variables": ["spco2"],
        #                      "tolerances": [0, 0.25, 0.25, 5],
        #                      "hasDepth": False,
        #                      "isClimatology": False
        #                      },
           "tblWOA_Climatology": {
                                  "variables": ["sea_water_temp_WOA_clim", "density_WOA_clim", "salinity_WOA_clim", "nitrate_WOA_clim", "phosphate_WOA_clim", "silicate_WOA_clim", "oxygen_WOA_clim", "AOU_WOA_clim", "o2sat_WOA_clim", "conductivity_WOA_clim"],
                                  "tolerances": [0, 0.75, 0.75, 5],
                                  "hasDepth": True,
                                  "isClimatology": True
                                  },
           "tblDarwin_Plankton_Climatology": {
                                  "variables": ["prokaryote_c01_darwin_clim", "prokaryote_c02_darwin_clim", "picoeukaryote_c03_darwin_clim", "picoeukaryote_c04_darwin_clim"],
                                  "tolerances": [0, 0.5, 0.5, 5],
                                  "hasDepth": True,
                                  "isClimatology": True
                                  },
           "tblDarwin_Nutrient_Climatology": {
                                  "variables": ["DIC_darwin_clim", "DOC_darwin_clim", "DOFe_darwin_clim", "DON_darwin_clim", "DOP_darwin_clim", "FeT_darwin_clim", "NH4_darwin_clim", "NO2_darwin_clim", "NO3_darwin_clim", "O2_darwin_clim", "PIC_darwin_clim", "PO4_darwin_clim", "POC_darwin_clim", "POFe_darwin_clim", "PON_darwin_clim", "POSi_darwin_clim", "SiO2_darwin_clim"],
                                  "tolerances": [0, 0.5, 0.5, 5],
                                  "hasDepth": True,
                                  "isClimatology": True
                                  },
           "tblArgo_MLD_Climatology": {
                                  "variables": ["mls_da_argo_clim", "mls_dt_argo_clim", "mlt_da_argo_clim", "mlt_dt_argo_clim", "mlpd_da_argo_clim", "mlpd_dt_argo_clim", "mld_da_mean_argo_clim", "mld_dt_mean_argo_clim"],
                                  "tolerances": [0, 0.75, 0.75, 5],
                                  "hasDepth": False,
                                  "isClimatology": True
                                  },
           "tblWOA_2018_1deg_Climatology": {
                                  "variables": ["A_an_clim", "A_mn_clim", "O_an_clim", "O_mn_clim", "si_an_clim", "si_mn_clim", "n_an_clim", "n_mn_clim", "p_an_clim", "p_mn_clim"],
                                  "tolerances": [0, 0.75, 0.75, 5],
                                  "hasDepth": True,
                                  "isClimatology": True
                                  },
           "tblWOA_2018_qrtdeg_Climatology": {
                                  "variables": ["C_an_clim", "C_mn_clim", "s_an_clim", "s_mn_clim", "t_an_clim", "t_mn_clim"],
                                  "tolerances": [0, 0.25, 0.25, 5],
                                  "hasDepth": True,
                                  "isClimatology": True
                                  },
           "tblWOA_2018_MLD_qrtdeg_Climatology": {
                                  "variables": ["M_an_clim", "M_mn_clim"],
                                  "tolerances": [0, 0.25, 0.25, 5],
                                  "hasDepth": False,
                                  "isClimatology": True
                                  },
           }
    return envs


def env_vars():
    """
    Reurns a list of environmental variables to be colocalized with
    species observations.
    """
    envs = environmental_datasets()
    vars = []
    for _, env in envs.items():
        vars += env["variables"]
    return vars

def only_climatology():
    return ["sea_water_temp_WOA_clim", "salinity_WOA_clim", "DIC_darwin_clim", "NH4_darwin_clim", "DON_darwin_clim"]

def surface_features(index: int):
    """
    Return list of features selected for abundance measurements
    at surface (depth < 10 m). These features are outcome
    of a multi-stage (round) RFE feature selection algorithm.
    """
    rounds = [
              ['lat', 'lon', 'depth', 'sst', 'chlor_a', 'adt', 'O2', 'POC', 'NO3', 'Si', 'PO4', 'NH4_darwin_clim', 'POSi_darwin_clim', 's_an_clim'],
              ['lat', 'lon', 'depth', 'sst_2', 'sst', 'O2_2', 'T/NH4', 'A/S', 'Si_2', 'A/C', 'A/P', 'PO', 'PO4Si', 'C/P'],
            #   only_climatology()
             ]
    return rounds[index]


def surface_and_depth_features(index: int):
    """
    Return list of features selected for abundance measurements
    at surface and lower depth values. These features are outcome
    of a multi-stage (round) RFE feature selection algorithm.
    """
    rounds = [
              ['lat', 'lon', 'depth', 'sea_water_temp_WOA_clim', 'conductivity_WOA_clim', 's_an_clim', 'O2', 'Si', 'CHL', 'PO4', 'NH4_darwin_clim', 'NO3_darwin_clim', 'DIC_darwin_clim', 'NO2_darwin_clim'],
              ['lat', 'lon', 'depth', 'sea_water_temp_WOA_clim', 'DIC_darwin_clim', 'O2', 'CHL', 'Si_2', 'NO2_darwin_clim', 's_an_clim_2', 'N/T', 'N/C', 'N/O', 'N/PO4', 'N/NH4'],
            #   only_climatology()
             ]
    return rounds[index]


def feature_engineering(df: pd.DataFrame, surface: bool):

    for c in df.columns:
        if c in TARGETS + ["lat", "lon", "depth", "table", "cruise"]:
            continue
        df[c + "_2"] = np.power(df[c], 2)
        if df[c].min() > 0:
            df[c + "_0.5"] = np.power(df[c], 0.5)
            df[c + "_log"] = np.log10(df[c])

    if surface:  # surface engineered features
        df["T/S"] = df["sst"] / df["s_an_clim"]
        df["N/T"] = df["NO3"] / df["sst"]
        df["T/NH4"] = df["sst"] / df["NH4_darwin_clim"]
        df["N/C"] = df["NO3"] / df["chlor_a"]
        df["C/P"] = df["chlor_a"] / df["POC"]
        df["N/O"] = df["NO3"] / df["O2"]
        df["A/C"] = df["adt"] / df["chlor_a"]
        df["A/S"] = df["adt"] / df["s_an_clim"]
        df["A/P"] = df["adt"] / df["POC"]
        df["N/A"] = df["NO3"] / df["adt"]

        df["TS"] = df["sst"] * df["s_an_clim"]
        df["TN"] = df["sst"] * df["NO3"]
        df["TNH4"] = df["sst"] * df["NH4_darwin_clim"]
        df["CN"] = df["chlor_a"] * df["NO3"]
        df["CA"] = df["chlor_a"] * df["adt"]
        df["NA"] = df["NO3"] * df["adt"]
        df["PA"] = df["POC"] * df["adt"]
        df["PO"] = df["POC"] * df["O2"]
    else:
        df["T/S"] = df["sea_water_temp_WOA_clim"] / df["s_an_clim"]
        df["N/T"] = df["NO3_darwin_clim"] / df["sea_water_temp_WOA_clim"]
        df["T/NH4"] = df["sea_water_temp_WOA_clim"] / df["NH4_darwin_clim"]
        df["N/C"] = df["NO3_darwin_clim"] / df["CHL"]
        df["N/NH4"] = df["NO3_darwin_clim"] / df["NH4_darwin_clim"]
        df["N/O"] = df["NO3_darwin_clim"] / df["O2"]
        df["N/D"] = df["NO3_darwin_clim"] / df["DIC_darwin_clim"]
        df["C/PO4"] = df["CHL"] / df["PO4"]
        df["D/PO4"] = df["DIC_darwin_clim"] / df["PO4"]
        df["N/PO4"] = df["NO3_darwin_clim"] / df["PO4"]

        df["TS"] = df["sea_water_temp_WOA_clim"] * df["s_an_clim"]
        df["TN"] = df["sea_water_temp_WOA_clim"] * df["NO3_darwin_clim"]
        df["TNH4"] = df["sea_water_temp_WOA_clim"] * df["NH4_darwin_clim"]
        df["CN"] = df["CHL"] * df["NO3_darwin_clim"]
        df["CD"] = df["CHL"] * df["DIC_darwin_clim"]
        df["NC"] = df["NO3_darwin_clim"] * df["conductivity_WOA_clim"]
        df["SNH4"] = df["s_an_clim"] * df["NH4_darwin_clim"]
        df["PNO2"] = df["PO4"] * df["NO2_darwin_clim"]

    df["O2Si"] = df["O2"] * df["Si"]
    df["PO4Si"] = df["PO4"] * df["Si"]
    df["PO4NH4"] = df["PO4"] * df["NH4_darwin_clim"]
    df["SiNH4"] = df["Si"] * df["NH4_darwin_clim"]
    return df
