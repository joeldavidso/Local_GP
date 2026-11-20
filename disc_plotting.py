import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import skink_hep as skplt
import os
import scipy.stats as stats

def X_HH(mh1, mh2):
    R = ((mh1 - 124) / (0.1 * mh1)) ** 2 + ((mh2 - 117) / (0.1 * mh2)) ** 2

    return np.sqrt(R)

def Get_Resolved_SR(batch, nj):

    pass_bool = np.array(batch["pass_resolved"])
    
    pass_bool = np.logical_and(pass_bool, 
                               np.array(batch["X_hh"]) < 1.6)

    pass_bool = np.logical_and(pass_bool,
                               np.array(batch["dEta_hh_abs"]) < 1.5)

    NJ_bool = None
    if nj == "4b":
        NJ_bool = np.array(batch["ntag"]) >= int(nj[0])
    else:
        NJ_bool = np.array(batch["ntag"]) == int(nj[0])

    return np.logical_and(NJ_bool, pass_bool)

def Get_Boosted_SR(batch, nj):

    pass_bool = np.array(batch["pass_boosted"])
    
    pass_bool = np.logical_and(pass_bool, 
                               np.array(batch["X_hh"]) < 1.5)

    # pass_bool = np.logical_and(pass_bool,
    #                            np.array(batch["dEta_hh_abs"]) < 1.5)

    NJ_bool = None
    if nj == "4b":
        NJ_bool = np.array(batch["ntag"]) >= int(nj[0])
    else:
        NJ_bool = np.array(batch["ntag"]) == int(nj[0])

    return np.logical_and(NJ_bool, pass_bool)



NJ = "4b"
filepath_base = "/eos/atlas/atlascerngroupdisk/phys-higp/BCQ/HH4b/bbbbarista_ntuples/20260509_v11_"+NJ+"_1P/dataframes/"
year_files = {
            "16": "combined_skim_ggFhh_chhh0p0_mc20a__Nominal.parquet",
            # "17": "combined_skim_ggFhh_chhh1p0_mc20d__Nominal.parquet",
            # "18": "combined_skim_ggFhh_chhh1p0_mc20e__Nominal.parquet",
            # "22": "combined_skim_ggFhh_chhh1p0_mc23a__Nominal.parquet",
            # "23": "combined_skim_ggFhh_chhh1p0_mc23d__Nominal.parquet",
            # "24": "combined_skim_ggFhh_chhh1p0_mc23e__Nominal.parquet",
            }

year_lumis = {"16": 3.2,
              "17": 33,
              "18": 44.3,
              "22": 31.6,
              "23": 29,
              "24": 108}

year_to_run = {"16": "run2",
               "17": "run2",
               "18": "run2",
               "22": "run3",
               "23": "run3",
               "24": "run3"}

run_cross_sections = {"run2": 30.8, "run3": 34.1}

total_resolved = 0
for year in year_files.keys():
    filepath = filepath_base + year_files[year]
    dataset = pq.ParquetFile(filepath)

    cols = ["pass_resolved", "pass_resolved_SR", "pass_resolved_CR",
            "m_h1", "m_h2", "dEta_hh_abs", "ntag", "X_hh", "mc_sf", "luminosity_resolved"
            # "pass_boosted", "pass_boosted_CR", "pass_boosted_SR_ggF", "pass_boosted_SR_VBF"
            ]

    for N, batch in enumerate(tqdm(dataset.iter_batches(batch_size = 10_000, columns = cols), desc = filepath[-80:], leave = False)):

        pass_arr = Get_Resolved_SR(batch, NJ)

        scaled_arr = np.array(batch["mc_sf"]) * year_lumis[year] * run_cross_sections[year_to_run[year]] * 0.58**2

        total_resolved += np.sum(scaled_arr[pass_arr])

        # pass_arr = Get_Boosted_SR(batch, NJ)

print(total_resolved)