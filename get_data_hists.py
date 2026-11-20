import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import skink_hep as skplt
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def Gridpoints(m1_bins, m2_bins):
    # Gridpoints for 2d mass plot
    gridpoints = []
    for m1bin in m1_bins[1]:
        for m2bin in m2_bins[1]:
            gridpoints.append([m1bin,m2bin])

    return np.array(gridpoints)


def Check_Location(filepath):    
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath

def get_bins(nbins = 20, xmins = [58.8, 55.5], xmaxs = [189.2, 178.5]):
    m1_bins, m2_bins = skplt.Utils.get_bins(xmins[0],xmaxs[0],nbins),skplt.Utils.get_bins(xmins[1],xmaxs[1],nbins)
    return m1_bins, m2_bins, nbins

###############################################################################################
# Edge Finding

def X_HH(mh1, mh2):
    R = ((mh1 - 124) / (0.1 * mh1)) ** 2 + ((mh2 - 117) / (0.1 * mh2)) ** 2

    return np.sqrt(R)

def R_CR(mh1, mh2):
    R = (mh1 - 1.05 * 124) ** 2 + (mh2 - 1.05 * 117) ** 2

    return np.sqrt(R)

def xhh_curve(x, y):
    term_x = (x - 124) / (0.1 * x)
    term_y = (y - 117) / (0.1 * y)

    return np.sqrt(term_x**2 + term_y**2) - 1.6

def rcr_curve(x, y):
    term_x = x - 124 * 1.05
    term_y = y - 117 * 1.05

    return np.sqrt(term_x**2 + term_y**2) - 45

def Contour_Plot(function, xbins, ybins):
    X, Y = np.meshgrid(xbins,ybins)
    plt.contour(X, Y, function(X, Y), levels = [0], colors = "red", linewidths = 2, linestyles = "--")


def calculate_cumulative_arc_length(points):
    segment_diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(segment_diffs**2, axis=1))

    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    return cumulative_lengths, np.sum(segment_lengths)


def edge_bool(mh1_bins, mh2_bins, edge):
    x_min, x_max = mh1_bins[0], mh1_bins[-1]
    y_min, y_max = mh2_bins[0], mh2_bins[-1]
    N_grid = 300  # Resolution for the grid

    X_grid, Y_grid = np.meshgrid(
        np.linspace(x_min, x_max, N_grid), np.linspace(y_min, y_max, N_grid)
    )

    # Calculate the value of F(x, y) for every point in the grid
    if edge == "XHH":
        Z_grid = xhh_curve(X_grid, Y_grid)
    elif edge == "RCR":
        Z_grid = rcr_curve(X_grid, Y_grid)

    fig, ax = plt.subplots()
    contour_set = ax.contour(X_grid, Y_grid, Z_grid, levels=[0], linewidths=0)
    plt.close(fig)  # We close the figure since we only need the data, not the plot

    paths = contour_set.allsegs[0]
    if not paths:
        print("Error: Could not find the contour F(x, y) = 0 within the defined grid.")
        exit()

    # Get vertices from the first path
    initial_points = paths[0]
    if not np.allclose(initial_points[0], initial_points[-1]):
        initial_points = np.vstack([initial_points, initial_points[0]])

    cumulative_s, total_L = calculate_cumulative_arc_length(initial_points)

    N_uniform = 10000  # Number of uniform points you want
    s_targets = np.linspace(
        0, total_L, N_uniform, endpoint=False
    )  # Exclude endpoint if loop

    # Interpolate X and Y coordinates as a function of the cumulative arc length 's'
    x_interp = interp1d(cumulative_s, initial_points[:, 0])
    y_interp = interp1d(cumulative_s, initial_points[:, 1])

    # Get the uniformly sampled coordinates
    x_uniform = x_interp(s_targets)
    y_uniform = y_interp(s_targets)

    uniform_points = np.array([x_uniform, y_uniform]).T

    return (
        np.histogram2d(
            uniform_points[:, 0], uniform_points[:, 1], bins=[mh1_bins, mh2_bins]
        )[0]
        > 0
    )

######################################################################################

def plot_hist2d(hist, m1_bins, m2_bins, NJ, mask, plotdir, plotname):

    tempgridpoints = Gridpoints(m1_bins, m2_bins)

    temphist = hist.flatten()
    temphist[~mask] = None

    hist2d = skplt.Hist2D.Hist2D(m1_bins, m2_bins, margins = False, cbar = True, 
                           xlabel = r"$m_{H1}$ [GeV]", ylabel = r"$m_{H2}$ [GeV]",
                           logz = False)
    hist2d.Set(tempgridpoints[:, 0], tempgridpoints[:,1], weights = temphist,
               zlabel = "No. "+str(NJ)+" Events")
    Contour_Plot(xhh_curve, tempgridpoints[:,0], tempgridpoints[:,1])
    hist2d.Text("ATLAS Internal", ypos = 0.9)
    hist2d.Plot(plotdir + plotname)


def Get_Data(filepath, m1_bins, m2_bins, NJ, year, mask, is_GP = False):

    print("Reading Data")

    # Grab the m1 m2 values of all events in the CR or SR
    arrays = Read_Files(filepath, m1_bins, m2_bins, NJ, is_GP)

    print("Binning Data")

    weights = arrays["weight"].flatten() if "weight" in arrays.keys() else None

    # Bin the data arrays into aingle 2d massplane hist 
    hist = np.histogram2d(arrays["m_h1"].flatten(), arrays["m_h2"].flatten(),
                          bins = [m1_bins[0], m2_bins[0]],
                          weights = weights
                          )[0]

    hist_coms = None
    hist_CR = None

    if is_GP:
        hist_uncs = np.histogram2d(arrays["m_h1"].flatten(), arrays["m_h2"].flatten(),
                                   bins = [m1_bins[0], m2_bins[0]],
                                   weights = arrays["weight_GP_up"].flatten() - weights
                                   )[0]

        CRmask = X_HH(arrays["m_h1"].flatten(), arrays["m_h2"].flatten()) > 1.6
        NSR = np.sum(~CRmask)
        print("No. "+str(NJ)+" SR predicted GP events: " + str(NSR))

    else:
        hist_uncs = np.sqrt(hist)

        # get centre of mass for edgebins and edgebin counts outside SR
        CRmask = X_HH(arrays["m_h1"].flatten(), arrays["m_h2"].flatten()) > 1.6
        hist_CR = np.histogram2d(arrays["m_h1"].flatten()[CRmask], arrays["m_h2"].flatten()[CRmask],
                                 bins = [m1_bins[0], m2_bins[0]],
                                 weights = weights
                                 )[0]

        hist_m1 = np.histogram2d(arrays["m_h1"].flatten()[CRmask], arrays["m_h2"].flatten()[CRmask],
                                 bins = [m1_bins[0], m2_bins[0]],
                                 weights = arrays["m_h1"].flatten()[CRmask]
                                 )[0]
        hist_m2 = np.histogram2d(arrays["m_h1"].flatten()[CRmask], arrays["m_h2"].flatten()[CRmask],
                                 bins = [m1_bins[0], m2_bins[0]],
                                 weights = arrays["m_h2"].flatten()[CRmask]
                                 )[0]

        hist_coms = np.stack([hist_m1.flatten()/hist_CR.flatten(), hist_m2.flatten()/hist_CR.flatten()],axis = 1)


        dEtamask = arrays["dEta_hh_abs"].flatten() < 1.5
        print("No. "+str(NJ)+" SR events (inclusive deta): " + str(np.sum(~CRmask)))
        print("No. "+str(NJ)+" SR events (deta < 1.5): " + str(np.sum(np.logical_and(~CRmask, dEtamask))))
        print("No. "+str(NJ)+" CR events (inclusive deta): " + str(np.sum(CRmask)))
        print("No. "+str(NJ)+" CR events (deta < 1.5): " + str(np.sum(np.logical_and(CRmask, dEtamask))))

    plotdir = Check_Location("plots/Counts/")
    plotdir = Check_Location(plotdir + str(NJ) + "/")
    plot_hist2d(hist, m1_bins, m2_bins, NJ, mask, plotdir, year)

    return hist, hist_uncs, hist_CR, hist_coms


def Read_Files(filepath, m1_bins, m2_bins, NJ, is_GP):

    arrays = {"m_h1": [],
              "m_h2": []}

    dataset = pq.ParquetFile(filepath)

    if is_GP:
        cols = ["m_h1", "m_h2", "weight_GP_mean", "weight_GP_up"]
        arrays["weight"] = []
        arrays["weight_GP_up"] = []
    else:
        cols = ["m_h1", "m_h2", "pass_resolved", "pass_vbf_sel", "ntag", "dEta_hh_abs"]
        arrays["dEta_hh_abs"] = []

    min_m1 = m1_bins[0][0]
    max_m1 = m1_bins[0][-1]
    min_m2 = m2_bins[0][0]
    max_m2 = m2_bins[0][-1]

    for N, batch in enumerate(tqdm(dataset.iter_batches(batch_size = 10_000, columns = cols), desc = filepath[-80:], leave = False)):

        if not is_GP:
            mask = Get_Cuts(batch, NJ)
        else:
            mask = np.ones_like(np.array(batch["m_h1"]), dtype = bool)

        m1_bool = np.logical_and(np.array(batch["m_h1"]) < max_m1, np.array(batch["m_h1"]) > min_m1)
        m2_bool = np.logical_and(np.array(batch["m_h2"]) < max_m2, np.array(batch["m_h2"]) > min_m2)

        mask = np.logical_and(mask, m1_bool)
        mask = np.logical_and(mask, m2_bool)

        for key in arrays.keys():
            if key == "weight" and is_GP:
                arrays[key].append(np.array(batch["weight_GP_mean"])[mask])
            else:
                arrays[key].append(np.array(batch[key])[mask])


    for key in arrays.keys():
        arrays[key] = np.concatenate(arrays[key])

    return arrays

def Get_Cuts(batch, nj):

    resolved_bool = np.array(batch["pass_resolved"])
    
    vbf_bool = np.logical_not(np.array(batch["pass_vbf_sel"]))

    pass_bool = np.logical_and(resolved_bool, vbf_bool)

    NJ_bool = None
    if nj == "4b":
        NJ_bool = np.array(batch["ntag"]) >= int(nj[0])
    else:
        NJ_bool = np.array(batch["ntag"]) == int(nj[0])

    return np.logical_and(NJ_bool, pass_bool)

def scale_down_xs(xs, m1_bins, m2_bins):

    min_m1 = m1_bins[0][0]
    max_m1 = m1_bins[0][-1]
    min_m2 = m2_bins[0][0]
    max_m2 = m2_bins[0][-1]

    xs[:,0] = (xs[:,0] - min_m1) / (max_m1 - min_m1)
    xs[:,1] = (xs[:,1] - min_m2) / (max_m2 - min_m2)

    return xs

def get_bin_densities(m1_bins, m2_bins, edgemask):
    m1s = np.random.uniform(m1_bins[0][0], m1_bins[0][-1], 100_000)
    m2s = np.random.uniform(m2_bins[0][0], m2_bins[0][-1], 100_000)
    xhhs = X_HH(m1s, m2s) > 1.6

    temphist = np.histogram2d(m1s[xhhs], m2s[xhhs], bins = [m1_bins[0], m2_bins[0]])[0].flatten()

    ref = np.mean(temphist[np.logical_and(temphist > 0, ~edgemask)])

    return temphist[edgemask]/ref

###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":

    m1_bins, m2_bins, nbins = get_bins()
    data = True
    GP = False

    # NJs = ["2b2j", "3b1f", "4b"]
    # NJs = ["3b1f", "4b"]
    NJs = ["4b"]

    for NJ in NJs:
        print("No. b-jets: "+NJ[:2]) 

        filepath_base = "/eos/atlas/atlascerngroupdisk/phys-higp/BCQ/HH4b/bbbbarista_ntuples/20260509_v11_2b_1P/dataframes/"
        year_files = {
                    "16": "combined_skim_data16__Nominal.parquet",
                    "17": "combined_skim_data17__Nominal.parquet",
                    "18": "combined_skim_data18__Nominal.parquet",
                    # "22": "combined_skim_data22__Nominal.parquet",
                    # "23": "combined_skim_data23__Nominal.parquet",
                    # "24": "combined_skim_data24__Nominal.parquet"
                    }



        filepath_base_gp = "/eos/atlas/atlascerngroupdisk/phys-higp/BCQ/HH4b/background_estimates/20260819_resolved_data24/"
        if NJ == "4b":
            filepath_base_gp += "data161718222324-gn277_"+NJ[:2]+"_sr_124_117_cr_x1.6_ggf_resolved_resolved_ggF_comb_allyears_finetune_"
        else:
            filepath_base_gp += "data161718222324-gn277_"+NJ[:2]+"_sr_124_117_cr_x1.6_ggf_resolved_resolved_ggF_comb_allyears_"+NJ+"_finetune_"

        year_files_gp = {
                        "16": "16/predictions/df_pred__pipe_16_target_16_50xGP_mean.parquet",
                        # "17": "17/predictions/df_pred__pipe_17_target_17_50xGP_mean.parquet",
                        # "18": "18/predictions/df_pred__pipe_18_target_18_50xGP_mean.parquet",
                        # "22": "22/predictions/df_pred__pipe_22_target_22_50xGP_mean.parquet",
                        # "23": "23/predictions/df_pred__pipe_23_target_23_50xGP_mean.parquet",
                        # "24": "24/predictions/df_pred__pipe_24_target_24_50xGP_mean.parquet"
                    }

        


        # Get array of bincentres
        gridpoints = Gridpoints(m1_bins, m2_bins)

        # Get bin mask for GP fit (CR bins not on SR edge)
        xhh_edgebool = edge_bool(m1_bins[0], m2_bins[0], "XHH").flatten()
        region_masks = {"CR": X_HH(gridpoints[:, 0], gridpoints[:, 1]) > 1.6,
                        "SR": X_HH(gridpoints[:, 0], gridpoints[:, 1]) < 1.6}

        fit_mask = np.logical_and(~xhh_edgebool, region_masks["CR"])

        np.save("hists/densities.npy", get_bin_densities(m1_bins, m2_bins, xhh_edgebool))

        np.save("hists/fit_mask.npy", fit_mask)
        np.save("hists/gridpoints.npy", gridpoints)
        np.save("hists/xhh_edgebool.npy", xhh_edgebool)

        rcr_edgebool = edge_bool(m1_bins[0], m2_bins[0], "RCR").flatten()
        rcr_mask = np.logical_and(R_CR(gridpoints[:,0], gridpoints[:, 1]) < 45, fit_mask)
        rcr_mask = np.logical_and(rcr_mask, ~rcr_edgebool)

        np.save("hists/rcr_mask.npy", rcr_mask)

        # # Get data histograms
        if data:
            print("Getting Data Hists!")
            for year in year_files.keys():
                print("YEAR: "+year)
                hist, hist_uncs, hist_CR, hist_coms = Get_Data(filepath_base + year_files[year],
                                                               m1_bins, m2_bins, NJ, year, fit_mask)

                np.save("hists/"+year+"_"+NJ+".npy", hist)
                np.save("hists/"+year+"_"+NJ+"_uncs.npy", hist_uncs)
                np.save("hists/"+year+"_"+NJ+"_CR.npy", hist_CR)
                np.save("hists/"+year+"_"+NJ+"_coms_CR.npy", hist_coms)

        # Get GP histograms
        if GP:
            print("Getting GP Hists!")
            for year in year_files_gp.keys():
                print("YEAR: "+year)
                hist, hist_uncs, _, __ = Get_Data(filepath_base_gp + year_files_gp[year],
                                                  m1_bins, m2_bins, NJ, year, fit_mask, is_GP = True)

                np.save("hists/"+year+"_"+NJ+"_GP.npy", hist)
                np.save("hists/"+year+"_"+NJ+"_GP_uncs.npy", hist_uncs)

        print("DONE!!!")