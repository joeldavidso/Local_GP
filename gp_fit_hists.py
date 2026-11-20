import numpy as np
import skink_hep as skplt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
import pickle
import matplotlib.pyplot as plt
import os
from get_data_hists import Gridpoints, get_bins


def Check_Location(filepath):    
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath


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


def scale_down_hist(hist, hist_uncs):
    mean = np.mean(hist)
    std = np.std(hist)
    hist_new = (hist - mean)/std
    hist_uncs_new = hist_uncs/std
    return mean, std, hist_new, hist_uncs_new


def scale_up_hist(hist, cov, mean, std):
    return (hist * std) + mean, cov * std**2

def apply_anscombe(hist):
    temphist = 2 * np.sqrt(hist + 3/8)
    return temphist

def undo_anscombe(hist, cov):
    temphist = (hist/2)**2 - 3/8

    tempcov = np.diag(hist/2) @ cov @ np.diag(hist/2)

    return temphist, tempcov

def scale_down_xs(xs, m1_bins, m2_bins):

    xs_temp = np.zeros_like(xs)

    min_m1 = m1_bins[0][0]
    max_m1 = m1_bins[0][-1]
    min_m2 = m2_bins[0][0]
    max_m2 = m2_bins[0][-1]

    xs_temp[:,0] = (xs[:,0] - min_m1) / (max_m1 - min_m1)
    xs_temp[:,1] = (xs[:,1] - min_m2) / (max_m2 - min_m2)

    return xs_temp


def scale_up_xs(xs, m1_bins, m2_bins):

    min_m1 = m1_bins[0][0]
    max_m1 = m1_bins[0][-1]
    min_m2 = m2_bins[0][0]
    max_m2 = m2_bins[0][-1]

    xs[:,0] = (xs[:,0]) * (max_m1 - min_m1) + min_m1
    xs[:,1] = (xs[:,1]) * (max_m2 - min_m2) + min_m2

    return xs


def fit_and_save_gp(year, NJ, xs, ys, ys_uncs, kernel, **kernel_kwargs):

    gpr = GaussianProcessRegressor(kernel = kernel(**kernel_kwargs), random_state = 0,
                                   alpha = ys_uncs**2, n_restarts_optimizer = 10).fit(xs, ys)

    pickle.dump(gpr, open("gp_fits/"+year+"_"+NJ+".sav", "wb"))

    return gpr


def predict_gp(year, NJ, xs):

    gpr = pickle.load(open("gp_fits/"+year+"_"+NJ+".sav", "rb"))

    print(xs.shape)

    means, cov = gpr.predict(xs, return_cov = True)

    return means, cov

def lengthscalescan(year, NJ, xs, ys, ys_uncs, m1_bins, m2_bins, L_opts, L_min, L_max, nbins, plotdir, plotname):

    params_arr = []
    lengthscales = skplt.Utils.get_bins(L_min, L_max, nbins)
    lengthscale_gridpoints = Gridpoints(lengthscales,lengthscales)

    for L in lengthscale_gridpoints:
        gpr = fit_and_save_gp(year, NJ, xs, ys, ys_uncs, RBF, length_scale = L, length_scale_bounds = "fixed")

        params_arr.append(n_params_effective(gpr.kernel_(xs), np.diag(gpr.alpha)))

    params_arr = np.array(params_arr)

    L_opts = scale_up_xs(np.array([L_opts]), m1_bins, m2_bins)[0]

    lengthscales = scale_up_xs(np.column_stack((lengthscales[0], lengthscales[0])), m1_bins, m2_bins)[:,0]
    lengthscales = skplt.Utils.get_bins(np.min(lengthscales), np.max(lengthscales), nbins)

    lengthscale_gridpoints = scale_up_xs(lengthscale_gridpoints, m1_bins, m2_bins)

    hist2d = skplt.Hist2D.Hist2D(lengthscales, lengthscales, margins = False, cbar = True,
                                 xlabel = r"$m_{H1}$ Lengthscale [GeV]", ylabel = r"$m_{H2}$ Lengthscale [GeV]",
                                 logz = True)
    hist2d.Set(lengthscale_gridpoints[:,0], lengthscale_gridpoints[:,1], weights = params_arr, zlabel = "Effective NDoF")
    plt.vlines(L_opts[0], np.min(lengthscales[0]), np.max(lengthscales[0]), linestyles = "--", colors = "#cc3300")
    plt.hlines(L_opts[1], np.min(lengthscales[0]), np.max(lengthscales[0]), linestyles = "--", colors = "#cc3300")
    plt.plot(L_opts[0], L_opts[1], markersize = 10, color = "#cc3300")
    hist2d.Text("ATLAS Internal", ypos = 0.9)

    hist2d.Plot(plotdir + plotname.replace(" ", "_") + "_"+year)


def pull_plots(x, x_cov, x_nparams, y, y_cov, y_nparams, gridpoints, bins, contours, plotdir, plotname, pullorunc = "pull", region = "CR"):

    m1_bins, m2_bins = bins

    # chi2 stuff
    residuals = x - y
    chi2 = residuals.T @ np.linalg.inv(x_cov + y_cov) @ residuals
    ndof = x.shape[0] - x_nparams
    chi2_unc = np.sqrt(2/ndof)

    values = None
    zlabel = None
    bins_1d = None
    if pullorunc == "pull":
        # pulls
        values = residuals/np.sqrt(np.diag(x_cov) + np.diag(y_cov))
        zlabel = r"$(Obs - Pred) / \sigma$"
        bins_1d = skplt.Utils.get_bins(-4, 4, 20)

        plot_truncated_chi2s(x_cov + y_cov, residuals,
                             x_cov, y_cov, 
                             plotdir, plotname, region)

    elif pullorunc == "unc":
        # uncs
        values = np.sqrt(np.diag(x_cov) / np.diag(y_cov))
        zlabel = r"$\sigma_{Pred} / \sigma_{Obs}$"
        bins_1d = skplt.Utils.get_bins(0.1, 1.5, 40)

    # 2D pull plot
    plotdirtemp = Check_Location(plotdir + "2D/")
    hist2d = skplt.Hist2D.Hist2D(m1_bins, m2_bins, margins = False, cbar = True,
                                 xlabel = r"$m_{H1}$", ylabel = r"$m_{H2}$")
    hist2d.Set(gridpoints[:, 0], gridpoints[:,1], weights = values, zlabel = zlabel)
    for contour in contours:
        Contour_Plot(contour, m1_bins[1], m2_bins[1])
    hist2d.ax_primary.grid(False)
    hist2d.Text("ATLAS Internal", ypos = 0.9)
    hist2d.Plot(plotdirtemp + plotname.replace(" ", "_"))

    # 1D pull plot
    plotdirtemp = Check_Location(plotdir + "1D/")
    hist1d = skplt.Hist1D.HistogramPlot(bins = bins_1d,
                                        xlabel = zlabel, ylabel = "No. Bins", plot_unc = False)
    hist1d.Add(values, label = plotname)
    hist1d.Text(r"$\chi^2/n_{dof}$: "+str(round(chi2/ndof, 2))+r" $\pm$ "+str(round(chi2_unc, 2)), 0.14, 0.76)
    hist1d.Text(r"$\mu$: "+str(round(np.mean(values), 2)), 0.14, 0.71)
    hist1d.Text(r"$\sigma$: "+str(round(np.std(values), 2)), 0.14, 0.66)
    hist1d.Text("ATLAS Internal")
    hist1d.Plot(plotdirtemp + plotname.replace(" ", "_"))


def marginal_plot(Fit_T, GP_T, Data_T, bins, plotdir, plotname, fit_label = "Local GP", GP_label = "Analysis GP", negy = False):

    labels = [r"$m_{H1}$", r"$m_{H2}$"]
    names = ["m_H1", "m_H2"]

    for I in range(2):

        Fit, Fit_uncs = marginalize(Fit_T[0], Fit_T[1], 1 - I)
        GP, GP_uncs = marginalize(GP_T[0], GP_T[1], 1 - I)
        Data, Data_uncs = marginalize(Data_T[0], Data_T[1], 1 - I)
        # Data_uncs = np.sqrt(Data)

        plot = skplt.Hist1D.HistogramPlot(bins = bins[I], xlabel = labels[I] + "[GeV]", ylabel = "No. Events",
                                          pull = True, n_pulls = 1, ratio = True, n_ratios = 1, negy = negy)
        plot.Add(bins[I][1], weights = Fit, uncs = Fit_uncs, label = fit_label, pull_plot = "Data", ratio_plot = "Data")
        plot.Add(bins[I][1], weights = GP, uncs = GP_uncs, label = GP_label,
                 linestyle = "--", pull_plot = "Data", ratio_plot = "Data")
        plot.Add(bins[I][1], weights = Data, uncs = Data_uncs, label = "Data",
                 linestyle = "", markersize = 10, reference = True, pull_plot = "Data", ratio_plot = "Data")
        plot.Text("ATLAS Internal")
        plot.Plot(plotdir + plotname.replace(" ", "_") + "_" + names[I])
        plt.close("all")
        plt.clf()

def marginalize(yields, uncs, index):

    yields_margin = np.sum(yields, axis = index)
    uncs_margin = np.sqrt(np.sum(uncs**2, axis = index))

    return yields_margin, uncs_margin

def array_to_CR(x, uncs, mask, nbins):

    x_temp = np.zeros_like(mask, dtype = x.dtype)
    uncs_temp = np.zeros_like(mask, dtype = uncs.dtype)

    x_temp[mask] = x
    uncs_temp[mask] = uncs

    x_temp = x_temp.reshape(nbins, nbins)
    uncs_temp = uncs_temp.reshape(nbins, nbins)

    return x_temp, uncs_temp


def impact_plots(xs, means, cov, bins, mask, plotdir, plotname):

    eigfs, e, V = get_eigfs(cov)

    plot_eigenvalues(e, plotdir, plotname)

    for count, var in enumerate(eigfs):

        impacts = (eigfs[-count-1] / means)
        impacts_temp = np.zeros_like(mask, dtype = impacts.dtype)
        impacts_temp[mask] = impacts
        impacts_temp[~mask] = np.nan

        histplot2d = skplt.Hist2D.Hist2D(bins[0], bins[1], margins = False, cbar = True,
                                xlabel = r"$m_{H1} [GeV]$", ylabel = r"$m_{H2} [GeV]$")
        histplot2d.Set(xs[:,0], xs[:,1], 
                       cmin = -np.max(np.abs(impacts_temp[mask])),
                       cmax = np.max(np.abs(impacts_temp[mask])),
                       weights = impacts_temp, zlabel = "Eigenvariation "+str(count)+" Impact")
        Contour_Plot(xhh_curve, xs[:,0], xs[:,1])
        histplot2d.ax_primary.grid(False)
        histplot2d.Text("ATLAS Internal", ypos = 0.9)
        histplot2d.Plot(plotdir + plotname+"_"+str(count))

        if count == 10:
            break

def get_eigfs(cov):
    # This funtion takes the covariance matrix
    # to return the eigenvariations

    e, V = np.linalg.eigh(cov)

    threshold = 1e-12
    e = np.where(e < threshold, 0.0, e)

    variation = V @ np.diag(np.sqrt(e))

    vars = []
    for i in range(variation.shape[0]):
        vars.append(variation[:,i])

    return vars, e, V

def n_params_effective(kernel, noise):

    H = kernel @ np.linalg.inv(kernel + noise)

    return np.linalg.trace(H)

def remove_row_column(arr, nbins_new, nbins_old):

    diff = nbins_new - nbins_old
    if diff >= 0:
        raise("UH what?")

    if arr.shape == (nbins_old, nbins_old):
        return arr[:diff, :diff]
    # elif arr.shape == (nbins_old**2,):
    #     return arr.reshape(20, 20)[:diff, :diff].flatten()
    elif arr.shape[0] == nbins_old**2:
        idx = np.arange(nbins_old**2)
        tempmask = (idx // nbins_old < nbins_old + diff) & (idx % nbins_old < nbins_old + diff)
        return(arr[tempmask])
    else:
        print(arr.shape)
        raise("wrong shape, is wrong!")


def plot_eigenvalues(eigenvalues, plotdir, plotname):

    eigenvalues = np.flip(eigenvalues)[:100]

    xs = np.arange(eigenvalues.shape[0])

    lineplot = skplt.Line.LinePlot(xs = xs, xlabel = "Eigenvalue Number", ylabel = "Eigenvalue Size",
                                   plot_unc = False, logy = False, ratio = True)
    lineplot.Add(ys = eigenvalues, label = "Eigenvalue Sizes", marker_size = 0, reference = True)
    lineplot.Add(ys = eigenvalues[0]*np.ones_like(eigenvalues), label = "Leading Eigenvalue Reference",
                 marker_size = 0, linestyle = "--", linecolour = "grey")
    lineplot.Plot(plotdir + plotname+"_eigenvalues")

def plot_truncated_chi2s(covariance, residuals, cov_gp, cov_noise, plotdir, plotname, region):
    
    # residuals = x - y
    # chi2 = residuals.T @ np.linalg.inv(x_cov + y_cov) @ residuals
    # ndof = x.shape[0] - x_nparams
    # chi2_unc = np.sqrt(2/ndof)

    C = cov_gp + cov_noise

    _, e, V = get_eigfs(C)

    # e_gp_projected = np.diag(V.T @ cov_gp @ V)
    
    # sort_idx = np.argsort(e)[::-1]
    # e = e[sort_idx]
    # V = V[:, sort_idx]
    
    # N_total = np.shape(e)[0]

    # chi2s = []
    # ndofs = []
    # chi2s_uncs = []
    # for k in range(N_total):

    #     retained_indices = slice(k, N_total)

    #     e_inv = 1/e[retained_indices]
    #     gamma = (V.T @ residuals)[retained_indices]

    #     chi2 = np.sum((gamma**2) @ e_inv)

    #     ndof =  N_total - k
    #     if region == "CR":
    #         ndof -= np.sum(e_gp_projected[retained_indices] * e_inv)
    #     chi2_unc = np.sqrt(2/ndof) if ndof > 0 else np.nan

    #     chi2s.append(chi2)
    #     ndofs.append(ndof)
    #     chi2s_uncs.append(chi2_unc)

    # e = np.flip(e)
    # V = np.flip(V)

    e_inv = 1/e
    N_total = e_inv.shape[0]

    chi2s = [residuals @ np.linalg.inv(C) @ residuals.T]
    chi2s_uncs = [np.sqrt(2/N_total)]
    ndofs = [N_total]

    for val in range(e.shape[0]):
        # e_inv[val] = 0
        # V[val] = 0
        # cov_inv = V.T @ np.diag(e_inv) @ V

        V_temp = V[val:]
        e_temp = e[val:]

        
        cov_temp = V_temp.T @ np.diag(1/e_temp) @ V_temp

        chi2s.append(residuals @ cov_temp @ residuals.T)
        chi2s_uncs.append(np.sqrt(2/(N_total)))
        ndofs.append(N_total)

    n_removed = np.arange(N_total + 1)
    chi2s = np.array(chi2s)
    ndofs = np.array(ndofs)
    chi2s_uncs = np.array(chi2s_uncs)

    lineplot = skplt.Line.LinePlot(xs = n_removed, xlabel = "No. Truncated Eigenvaraitaions", ylabel = r"$\chi^2/n_{dof}$", ratio = True)
    lineplot.Add(ys = chi2s / ndofs, label = r"$chi^2s$", reference = True, 
                 uncs = chi2s_uncs, marker_size = 0)
    lineplot.Add(ys = np.ones_like(chi2s) * chi2s[0] / ndofs[0], label = r"Full $chi^2$",
                 uncs = np.ones_like(chi2s) * chi2s_uncs[0],
                 linestyle = "--", linecolour = "grey", marker_size = 0)
    lineplot.Plot(plotdir + plotname+"_truncated_chis")
    print(plotdir + plotname+"_truncated_chis")



##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
if __name__ == "__main__":


    # from get_data_hists import nbins, m1_bins, m2_bins

    # mask = np.logical_and(xhh_mask, rcr_mask)

    # NJs = ["3b1f", "4b"]

    NJs = ["3b1f"]

    years = [
            "16", 
            "17",
            "18",
            "22",
            "23",
            "24"
            ]
    scalescan = False
    anscombe = True
    edge_binning = False
    region = "SR"

    for NJ in NJs:
        print("No. b-jets: "+NJ[:2]) 

        for year in years:

            m1_bins, m2_bins, nbins = get_bins()

            xhh_mask = np.load("hists/fit_mask.npy")
            rcr_mask = np.load("hists/rcr_mask.npy")
            xhh_edgebool = np.load("hists/xhh_edgebool.npy")   

            gridpoints = np.load("hists/gridpoints.npy")
            if edge_binning:
                coms = np.load("hists/"+year+"_"+NJ+"_coms_CR.npy")
                gridpoints[xhh_edgebool] = coms[xhh_edgebool]

            gridpoints_scaled = scale_down_xs(gridpoints, m1_bins, m2_bins)

            overall_CR_mask = xhh_mask
            # overall_CR_mask = np.logical_or(xhh_mask, xhh_edgebool)

            xhh_mask_temp = overall_CR_mask

            mask = overall_CR_mask
            if edge_binning:
                mask = np.logical_or(mask, xhh_edgebool)

            print("YEAR: "+year)

            hist_gp = np.load("hists/"+year+"_"+NJ+"_GP.npy")
            hist_uncs_gp = np.load("hists/"+year+"_"+NJ+"_GP_uncs.npy")

            hist = np.load("hists/"+year+"_"+NJ+".npy")
            hist_uncs = np.load("hists/"+year+"_"+NJ+"_uncs.npy")

            hist_new = hist.flatten()
            hist_uncs_new = hist_uncs.flatten()

            if edge_binning:
                cr_counts = np.load("hists/"+year+"_"+NJ+"_CR.npy")
                hist_new[xhh_edgebool] = cr_counts.flatten()[xhh_edgebool]/np.load("hists/densities.npy")
                hist_uncs_new[xhh_edgebool] = np.sqrt(cr_counts.flatten()[xhh_edgebool])/np.load("hists/densities.npy")

            if anscombe:
                hist_new = apply_anscombe(hist_new)
                hist_uncs_new = np.ones_like(hist_uncs_new)

            shift, scale, hist_new, hist_uncs_new = scale_down_hist(hist_new.flatten(), hist_uncs_new.flatten())


            np.save("hists/gp_scale_"+year+".npy", scale)
            np.save("hists/gp_shift_"+year+".npy", shift)

            plotdirbase = Check_Location("plots/GP_vs_GP/")
            plotdirbase = Check_Location(plotdirbase + NJ + "/")

            gpr = fit_and_save_gp(year, NJ, gridpoints_scaled[mask],
                                  hist_new[mask], hist_uncs_new[mask],
                                  RBF, length_scale = [0.5, 0.5], length_scale_bounds = (1e-12, 3)
                                  )
            L_opts = gpr.kernel_.length_scale

            if scalescan:
                # Do lengthscale scan before saving and fitting (not in that order) the final Local GP
                lengthscalescan(year, NJ, gridpoints_scaled[mask], hist_new[mask], hist_uncs_new[mask], m1_bins, m2_bins, L_opts,
                                0.1, 1, 10, plotdir = Check_Location(plotdirbase + "lengthscale/"), plotname = "Local GP")

                gpr = fit_and_save_gp(year, NJ, gridpoints_scaled[mask], 
                                      hist_new[mask], hist_uncs_new[mask],
                                      RBF, length_scale = [0.5, 0.5], length_scale_bounds = (1e-12, 3)
                                      )

            pred_mask = None
            if region == "CR":
                pred_mask = xhh_mask
            elif region == "SR":
                pred_mask = ~xhh_mask

            means, cov = predict_gp(year, NJ, gridpoints_scaled[pred_mask])

            nparams = n_params_effective(gpr.kernel_(gridpoints_scaled[mask]), np.diag(gpr.alpha))
            print("Fit Effective No. Params: ")
            print(nparams)

            if anscombe:

                means, cov = scale_up_hist(means, cov, shift, scale)
                means, cov = undo_anscombe(means, cov)
                

                print("Length Scales: ")
                print(np.array([gpr.kernel_.length_scale])[0])
                print("Scaled: ")
                print(scale_up_xs(np.array([gpr.kernel_.length_scale]), m1_bins, m2_bins)[0] - np.array([58.8, 55.5]))

                print("Scale: ")
                print(scale)
                print("Shift: ")
                print(shift)

                np.save("hists/gp_nparams_"+year+".npy", nparams)

                if region != "CR":
                    nparams = 0

            # pull plots and unc plots
                for value in ["pull", "unc"]:

                    plotdir = Check_Location(plotdirbase + value + "s/")
                    plotdir = Check_Location(plotdir + year +"/")
                    # Fit vs Data
                    pull_plots(means, cov, nparams, 
                               hist.flatten()[pred_mask], np.diag(hist_uncs.flatten()[pred_mask])**2, 0,
                               gridpoints[pred_mask], (m1_bins, m2_bins), [],
                               plotdir, "Local GP vs Data ", value, region)
                    # GP vs Data
                    pull_plots(hist_gp.flatten()[pred_mask], np.diag(hist_uncs_gp.flatten()[pred_mask])**2, 0,
                            hist.flatten()[pred_mask], np.diag(hist_uncs.flatten()[pred_mask])**2, 0,
                            gridpoints[pred_mask], (m1_bins, m2_bins), [],
                            plotdir,"Analysis GP vs Data", value, region)
                    # Fit vs GP
                    pull_plots(means, cov, nparams,
                            hist_gp.flatten()[pred_mask], np.diag(hist_uncs_gp.flatten()[pred_mask])**2, 0,
                            gridpoints[pred_mask], (m1_bins, m2_bins), [],
                            plotdir,"Local GP vs Analysis GP", value, region)


            # Marginal plots
                plotdir = Check_Location(plotdirbase+"marginals/")
                plotdir = Check_Location(plotdir + year +"/")
                # Fit vs GP vs Data
                means_temp, means_uncs_temp = array_to_CR(means, np.sqrt(np.diag(cov)), pred_mask, nbins)
                hist_gp_temp, hist_uncs_gp_temp = array_to_CR(hist_gp.flatten()[pred_mask], hist_uncs_gp.flatten()[pred_mask], pred_mask, nbins)
                hist_temp, hist_uncs_temp = array_to_CR(hist.flatten()[pred_mask], hist_uncs.flatten()[pred_mask], pred_mask, nbins)
                hist_uncs_temp = np.sqrt(hist_temp)

                marginal_plot([means_temp, means_uncs_temp],
                            [hist_gp_temp, hist_uncs_gp_temp],
                            [hist_temp, hist_uncs_temp],
                            [m1_bins, m2_bins], plotdir, "Local GP v GP v DATA")

            # Impact plots
                plotdir = Check_Location(plotdirbase+"impacts/")
                plotdir = Check_Location(plotdir + year +"/")
                # Fit

                # predict full massplane
                if NJ == "4b":
                    massplane_mask = xhh_mask_temp
                elif NJ == "3b1f":
                    massplane_mask = np.ones_like(mask, dtype = bool)


                means, cov = predict_gp(year, NJ, gridpoints_scaled[massplane_mask])
                means, cov = scale_up_hist(means, cov, shift, scale)

                plotdirtemp = Check_Location(plotdir + "Massplane/")
                impact_plots(gridpoints, means, cov, [m1_bins, m2_bins], massplane_mask,
                            plotdirtemp, "Local GP Impact Full Massplane")

                sr_mask = ~xhh_mask_temp

                means, cov = predict_gp(year, NJ, gridpoints_scaled[sr_mask])
                means, cov = scale_up_hist(means, cov, shift, scale)

                plotdirtemp = Check_Location(plotdir + "SR/")
                impact_plots(gridpoints, means, cov, [m1_bins, m2_bins], sr_mask,
                            plotdirtemp, "Local GP Impact SR")

                print("-----")
