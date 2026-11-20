from get_data_hists import get_bins
from scipy.optimize import curve_fit
from gp_fit_hists import marginalize
import numpy as np
import skink_hep as skplt
from gp_fit_hists import scale_down_xs, apply_anscombe, undo_anscombe, scale_up_xs, scale_up_hist, scale_down_hist, Check_Location, array_to_CR
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
import scipy.stats as scps
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

m1_bins, m2_bins, nbins = get_bins()

def exp_func(m, A, lam):
    return A * np.exp(-lam * m)

def solve_exp_decay(x1, F_x1, x2, F_x2):

    lam = (np.log(F_x1) - np.log(F_x2))/(x2 - x1)

    A = F_x1 * np.exp(lam * x1)

    return A, lam

def get_falling(n_events, m_min, m_max, lam_fit):

    u = np.random.uniform(0, 1, n_events)

    term_min = np.exp(-lam_fit * m_min)
    term_max = np.exp(-lam_fit * m_max)

    m_falling = -1.0 / lam_fit * np.log(term_min - u * (term_min - term_max))

    return m_falling

def marginal_plot(obs_T, pred_T, pred_res_T, pred_res_only_T, bins, plotdir, plotname, region, process, negy = False):

    labels = [r"$m_{H1}$", r"$m_{H2}$"]
    names = ["m_H1", "m_H2"]

    ratiolabel = "Data" if region == "CR" else "GP"
    pulllabel = ratiolabel

    for I in range(2):

        axis = I - 1

        def get_marg_values_and_uncs(values, uncs_or_cov):
            # If uncs_or_cov is a full covariance matrix (2D of size N_active x N_active)
            if uncs_or_cov.ndim == 2 and uncs_or_cov.shape[0] > values.shape[0]:
                nx, ny = values.shape
                cov_2d = uncs_or_cov
                
                # Build summation projection matrix C for marginalization
                if axis == 0: # marginalize over axis 0 -> result shape (ny,)
                    C = np.zeros((ny, nx * ny))
                    for j in range(ny):
                        for i in range(nx):
                            C[j, i * ny + j] = 1.0
                else: # marginalize over axis 1 -> result shape (nx,)
                    C = np.zeros((nx, nx * ny))
                    for i in range(nx):
                        for j in range(ny):
                            C[i, i * ny + j] = 1.0
                            
                marg_vals = values.sum(axis=axis)
                marg_cov = C @ cov_2d @ C.T
                marg_uncs = np.sqrt(np.diag(marg_cov))
                return marg_vals, marg_uncs
            else:
                # Fallback to standard bin-by-bin marginalize function from gp_fit_hists
                return marginalize(values, uncs_or_cov, axis)

        # Use the covariance-aware marginalization helper exclusively
        Obs, Obs_uncs = get_marg_values_and_uncs(obs_T[0], obs_T[1])
        Fit, Fit_uncs = get_marg_values_and_uncs(pred_T[0], pred_T[1])
        Fit_res, Fit_res_uncs = get_marg_values_and_uncs(pred_res_T[0], pred_res_T[1])
        Fit_res_only, Fit_res_only_uncs = get_marg_values_and_uncs(pred_res_only_T[0], pred_res_only_T[1])

        # if region == "SR":
        #     Fit_res_uncs *= 0
        Fit_res_uncs *= 0

        plot = skplt.Hist1D.HistogramPlot(bins = bins[I], xlabel = labels[I] + "[GeV]", ylabel = "No. 20"+year+" Events",
                                          pull = True, ratio = True, n_ratios = 1, n_pulls = 1, negy = negy)
        if region == "CR":
            plot.Add(bins[I][1], weights = Obs, uncs = Obs_uncs, label = "Data", ratio_plot = ratiolabel, pull_plot = pulllabel, reference = True)
        plot.Add(bins[I][1], weights = Fit, uncs = Fit_uncs, label = "GP (Fit on Data)", ratio_plot = ratiolabel, pull_plot = pulllabel, reference = (region == "SR"))
        plot.Add(bins[I][1], weights = Fit_res, uncs = Fit_res_uncs, label = "("+process+") + GP (Fit on Data - "+process+")", ratio_plot = ratiolabel, pull_plot = pulllabel)
        if region == "CR":
            plot.Add(bins[I][1], weights = Fit_res_only, uncs = Fit_res_only_uncs, label = "GP (Fit on Data - "+process+")",
                    reference = False, ratio_plot = ratiolabel, pull_plot = pulllabel)
        plot.ax_primary.set_xlabel("")
        plot.Plot(plotdir + plotname.replace(" ", "_") + "_" + names[I])
        plt.close("all")
        plt.clf()


def get_resonant_events(process, n_expected_events, hist, m1_bins, m2_bins, scaling = 1, n_sample = 1_000_000):

    gauss_values = {"Z": {"mean": 91, "spread": 15},
                    "H": {"mean": 125, "spread": 20}}

    # split process name so we know what features we need 
    proc_1 = process[0]
    proc_2 = process[1:]

    # get first feature, always gaussian
    feature_events_1_m1 = np.random.normal(loc = gauss_values[proc_1]["mean"], scale = gauss_values[proc_1]["spread"], size = n_sample)
    feature_events_1_m2 = np.random.normal(loc = gauss_values[proc_1]["mean"], scale = gauss_values[proc_1]["spread"], size = n_sample)

    # get second feature, either gaussian or exponential decay
    if proc_2 == "bb": # <------ Exponential Decay
    
        marg_hist_m1, _ = marginalize(hist, np.sqrt(hist), 1)
        marg_hist_m2, _ = marginalize(hist, np.sqrt(hist), 0)

        A_m1, lam_m1 = solve_exp_decay(m1_bins[1][0], marg_hist_m1[0], m1_bins[1][-1], marg_hist_m1[-1])
        A_m2, lam_m2 = solve_exp_decay(m2_bins[1][0], marg_hist_m2[0], m2_bins[1][-1], marg_hist_m2[-1])

        feature_events_2_m1 = get_falling(n_samples, m1_bins[0][0], m1_bins[0][-1], lam_m1)
        feature_events_2_m2 = get_falling(n_samples, m2_bins[0][0], m2_bins[0][-1], lam_m2)

    else:
        feature_events_2_m1 = np.random.normal(loc = gauss_values[proc_2]["mean"], scale = gauss_values[proc_2]["spread"], size = n_sample)
        feature_events_2_m2 = np.random.normal(loc = gauss_values[proc_2]["mean"], scale = gauss_values[proc_2]["spread"], size = n_sample)

    dt = np.dtype([("m1", feature_events_1_m1.dtype), ("m2", feature_events_1_m2.dtype), ("weight", feature_events_1_m1.dtype)])
    resonant_events = np.empty(2*n_samples, dtype=dt)

    resonant_events["m1"] = np.concatenate((feature_events_2_m1, feature_events_1_m1))
    resonant_events["m2"] = np.concatenate((feature_events_1_m1, feature_events_2_m2))

    weights = np.ones_like(resonant_events["m1"]) * (n_expected_events/(2*n_samples))

    resonant_events["weight"] = weights * res_bkgs_scaling

    return resonant_events

def calc_chi2_binned(obs, pred, cov, ndeff):
    # ndeff is the effective degrees of freedom of the GP

    residuals = obs - pred

    chi2 = residuals.T @ np.linalg.inv(cov + np.diag(pred)) @ residuals
    ndof = np.shape(residuals)[0] - ndeff
    chi2_unc = np.sqrt(2 / ndof)

    return chi2 / ndof, chi2_unc, ndof


def GP_yield_and_unc(pred, cov):

    mu = np.asarray(pred, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)

    total_yield = float(np.sum(mu))
    variance = float(np.sum(cov))  # same as np.ones_like(mu) @ cov @ np.ones_like(mu)
    uncertainty = float(np.sqrt(max(variance, 0.0)))

    return total_yield, uncertainty


# getdata hist and masks
NJ = "4b"

anscombe = True

luminosity_years = {"16": 35.6, "17": 46.9, "18": 60.6,
                    "22": 41, "23": 31.8, "24": 110}

luminosity_runs = {"run2": luminosity_years["16"] + luminosity_years["17"] + luminosity_years["18"],
                   "run3": luminosity_years["22"] + luminosity_years["23"] + luminosity_years["24"]}

expected_resonant_events = {"Zbb": {"run2": 8788 ,
                                    "run3": 20505},
                            "Hbb": {"run2": 1000 ,
                                    "run3": 2333 },
                            "ZZ":  {"run2": 848  ,
                                    "run3": 1980 },
                            "ZH":  {"run2": 236  ,
                                    "run3": 552  }}


def year_to_run(year):
    if year in ["16","17","18"]:
        return "run2"
    elif year in ["22","23","24"]:
        return "run3"
    else:
        raise("Unkown Year Given !!!!")

# can be Zbb or Hbb
inclusive_label = "Res. Bkgs."
process_names = ["Zbb", "Hbb", "ZZ", "ZH"]
processes = [inclusive_label]
years = ["16","17","18","22","23","24"]

res_bkgs_scaling = 0.5

for process in processes:
    print("------------------------------------------------------------------------")
    print(process+":")
    print("------------------------------------------------------------------------")
    for year in years:
        print(year+":")
        print("------------------------------------------")


        run = year_to_run(year)
        n_samples = 1_000_000

        plotdirbase = Check_Location("plots/resonant_"+process+"/")
        plotdirbase = Check_Location(plotdirbase+year+"/")

        xhh_mask = np.load("hists/fit_mask.npy")
        rcr_mask = np.load("hists/rcr_mask.npy")
        xhh_edgebool = np.load("hists/xhh_edgebool.npy")   

        gridpoints = np.load("hists/gridpoints.npy")


        hist = np.load("hists/"+year+"_"+NJ+".npy")
        hist_uncs = np.load("hists/"+year+"_"+NJ+"_uncs.npy")

        print("No. "+str(NJ)+" observed "+str(year)+" events:")
        print(np.sum(hist.flatten()[xhh_mask]))

        marg_hist_m1, _ = marginalize(hist, hist_uncs, 1)
        marg_hist_m2, _ = marginalize(hist, hist_uncs, 0)
        if process == inclusive_label:
            hist_resonant = np.zeros((20,20))
            for proc in process_names:
                n_exp_events = (luminosity_years[year]/luminosity_runs[run]) * expected_resonant_events[proc][run]
                resonant_events = get_resonant_events(proc, n_exp_events, hist, m1_bins, m2_bins, res_bkgs_scaling)
                hist_resonant_temp, xedges, yedges = np.histogram2d(resonant_events["m1"], resonant_events["m2"],
                                                                    bins = [m1_bins[0], m2_bins[0]],
                                                                    weights = resonant_events["weight"])
                hist_resonant += hist_resonant_temp

        else:
            n_exp_events = (luminosity_years[year]/luminosity_runs[run]) * expected_resonant_events[process][run]
            resonant_events = get_resonant_events(process, n_exp_events, hist, m1_bins, m2_bins, res_bkgs_scaling)

            hist_resonant, xedges, yedges = np.histogram2d(resonant_events["m1"], resonant_events["m2"],
                                                        bins = [m1_bins[0], m2_bins[0]],
                                                        weights = resonant_events["weight"])

        # plot resonant events
        plt.figure(figsize=(6, 5))
        mesh = plt.pcolormesh(xedges, yedges, hist_resonant.T, cmap='viridis', edgecolors='none')
        plt.grid(False)
        plt.colorbar(mesh, label='Expected No. 20'+str(year)+' Events')
        plt.xlabel('mh1 [GeV]')
        plt.ylabel('mh2 [GeV]')
        plt.title('Resonant '+process+' events')
        plt.savefig(plotdirbase+"resonant_2d.png")
        plt.savefig(plotdirbase+"resonant_2d.pdf")
        plt.clf()
        plt.close()


        # plot ratio data / (data - resonant) 

        ratiohist = hist_resonant / hist
        ratiohist[~(xhh_mask.reshape(20,20))] = np.nan

        plt.figure(figsize=(6, 5))
        mesh = plt.pcolormesh(xedges, yedges, ratiohist.T, cmap='viridis', edgecolors='none')
        plt.grid(False)
        plt.colorbar(mesh, label=process+'/Data')
        plt.xlabel('mh1 [GeV]')
        plt.ylabel('mh2 [GeV]')
        plt.title('20'+year)
        plt.savefig(plotdirbase+"resonant_2d_ratio.png")
        plt.savefig(plotdirbase+"resonant_2d_ratio.pdf")
        plt.clf()
        plt.close()

        hist_resonant_m1, _ = marginalize(hist_resonant, np.sqrt(hist_resonant), 1)
        hist_resonant_m2, _ = marginalize(hist_resonant, np.sqrt(hist_resonant), 0)

        plot = skplt.Hist1D.HistogramPlot(bins = m1_bins, xlabel = r"$m_{H1}$"+"[GeV]", ylabel = "No. Events", ratio = True)
        plot.Add(m1_bins[1], weights = marg_hist_m1, uncs = np.sqrt(marg_hist_m1), label = "Data", reference = True)
        plot.Add(m1_bins[1], weights = marg_hist_m1 - hist_resonant_m1,
                uncs = np.sqrt(marg_hist_m1 - hist_resonant_m1), label = "Data - ("+process+")",
                linestyle = "--")
        plot.Add(m1_bins[1], weights = hist_resonant_m1, uncs = np.sqrt(hist_resonant_m1), label = "exp "+process+" events", addthis = False)
        plot.ax_primary.set_xlabel("")
        plot.Plot(plotdirbase+"resonant_marginal_m1")


        plot = skplt.Hist1D.HistogramPlot(bins = m2_bins, xlabel = r"$m_{H2}$"+"[GeV]", ylabel = "No. Events", ratio = True)
        plot.Add(m2_bins[1], weights = marg_hist_m2, uncs = np.sqrt(marg_hist_m2), label = "Data", reference = True)
        plot.Add(m2_bins[1], weights = marg_hist_m2 - hist_resonant_m2,
                uncs = np.sqrt(marg_hist_m2 - hist_resonant_m2), label = "Data - ("+process+")",
                linestyle = "--")
        plot.Add(m2_bins[1], weights = hist_resonant_m2, uncs = np.sqrt(hist_resonant_m2), label = "exp "+process+" events", addthis = False)
        plot.ax_primary.set_xlabel("")
        plot.Plot(plotdirbase+"resonant_marginal_m2")


        ###############################################################################

        # get gp fits on the data

        # data = hist
        # resonant events =  hist_resonant

        gridpoints_scaled = scale_down_xs(gridpoints, m1_bins, m2_bins)

        mask = xhh_mask
        masks = {"CR": xhh_mask, "SR": ~xhh_mask}

        def get_hist_fit(hist, hist_uncs, anscombe):

            hist_fit = hist.flatten()
            hist_uncs_fit = np.sqrt(hist_fit)

            if anscombe:
                hist_fit = apply_anscombe(hist_fit)
                hist_uncs_fit = np.ones_like(hist_uncs_fit)

            shift, scale, hist_fit, hist_uncs_fit = scale_down_hist(hist_fit.flatten(), hist_uncs_fit.flatten())

            return shift, scale, hist_fit, hist_uncs_fit

        shift, scale, hist_fit, hist_uncs_fit = get_hist_fit(hist, np.sqrt(hist), anscombe)
        hist_diff = hist - hist_resonant
        shift_res, scale_res, hist_fit_res, hist_uncs_fit_res = get_hist_fit(hist_diff, np.sqrt(hist_diff), anscombe)

        #fit gps
        gpr = GaussianProcessRegressor(kernel = RBF(length_scale = [0.5, 0.5], length_scale_bounds = (1e-12, 3)),
                                    random_state = 0,
                                    alpha = hist_uncs_fit[masks["CR"]]**2,
                                    n_restarts_optimizer = 10).fit(gridpoints_scaled[masks["CR"]], hist_fit[masks["CR"]])

        gpr_res = GaussianProcessRegressor(kernel = RBF(length_scale = [0.5, 0.5], length_scale_bounds = (1e-12, 3)),
                                        random_state = 0,
                                        alpha = hist_uncs_fit_res[masks["CR"]]**2,
                                        n_restarts_optimizer = 10).fit(gridpoints_scaled[masks["CR"]], hist_fit_res[masks["CR"]])

        L_opts = gpr.kernel_.length_scale
        L_opts = scale_up_xs(np.array([L_opts]), m1_bins, m2_bins)[0] - [m1_bins[1][0], m2_bins[1][0]]

        L_opts_res = gpr_res.kernel_.length_scale
        L_opts_res = scale_up_xs(np.array([L_opts_res]), m1_bins, m2_bins)[0] - [m1_bins[1][0], m2_bins[1][0]]

        print("Lengthscales (data fit) [GeV]:")
        print(L_opts[0])
        print(L_opts[1])

        print("Lengthscales (data - "+process+" fit)[GeV]:")
        print(L_opts_res[0])
        print(L_opts_res[1])

        for mask in masks.keys():

            #predict gps
            pred, cov = gpr.predict(gridpoints_scaled[masks[mask]], return_cov = True)
            pred_res, cov_res = gpr_res.predict(gridpoints_scaled[masks[mask]], return_cov = True)


            if anscombe:

                pred, cov = scale_up_hist(pred, cov, shift, scale)
                pred, cov = undo_anscombe(pred, cov)

                pred_res, cov_res = scale_up_hist(pred_res, cov_res, shift_res, scale_res)
                pred_res, cov_res = undo_anscombe(pred_res, cov_res)
                
            # plot the marginal predictions

            pred_temp, pred_uncs_temp = array_to_CR(pred, np.sqrt(np.diag(cov)), masks[mask], nbins)
            pred_res_temp, pred_res_uncs_temp = array_to_CR(pred_res, np.sqrt(np.diag(cov_res)), masks[mask], nbins)

            hist_temp, hist_uncs_temp = array_to_CR(hist.flatten()[masks[mask]], hist_uncs.flatten()[masks[mask]], masks[mask], nbins)

            hist_resonant_temp = hist_resonant.flatten()
            hist_resonant_temp[~masks[mask]] = 0
            hist_resonant_temp = hist_resonant_temp.reshape(nbins,nbins)

            # get full covariances
            mask_flat = masks[mask].flatten()
            n_total = nbins * nbins

            cov_full = np.zeros((n_total, n_total))
            cov_full[np.ix_(mask_flat, mask_flat)] = cov

            cov_res_full = np.zeros((n_total, n_total))
            cov_res_full[np.ix_(mask_flat, mask_flat)] = cov_res



            # get chi2 and p-value

            temp_obs = pred_res + hist_resonant.flatten()[masks[mask]]
            reduced_chi2, chi2_unc, ndof = calc_chi2_binned(temp_obs, pred, cov, 0)
            p_value = scps.chi2.sf(reduced_chi2 * ndof, ndof)
            print(mask+":")
            print("Reduced Chi2: "+str(round(reduced_chi2,2)))
            print("P-Value: "+str(round(p_value,2)))

            yield_total, yield_unc = GP_yield_and_unc(pred, cov)
            yield_total_res, yield_unc_res = GP_yield_and_unc(pred_res, cov_res)
            yield_total_res += np.sum(hist_resonant.flatten()[masks[mask]])
            yield_unc_res = np.sqrt(yield_unc_res**2 + np.sum(hist_resonant.flatten()[masks[mask]]))
            print("Estimated GP (Data) Yield: "+str(int(yield_total)) +" +/- "+str(round(yield_unc,2)))
            print("Estimated Res. Bkgs. + GP (Data - Res. Bkgs.) Yield: "+str(int(yield_total_res)) +" +/- "+str(round(yield_unc_res,2)))

            marginal_plot([hist_temp, hist_uncs_temp],
                          [pred_temp, cov_full],
                          [pred_res_temp + hist_resonant_temp, cov_res_full],
                          [pred_res_temp, cov_res_full],
                          [m1_bins, m2_bins],
                          plotdirbase, process + " GP ("+mask+")with vs without",
                          mask,
                          process
                         )

            if mask == "CR" or NJ != "4b":
                # plot ratio of data to GP preds
                hist_ratio = hist_temp / pred_temp
                plt.figure(figsize=(6, 5))
                mesh = plt.pcolormesh(xedges, yedges, hist_ratio.T, cmap='viridis', edgecolors='none')
                plt.grid(False)
                plt.colorbar(mesh, label='Data / GP (Fit on Data)')
                plt.xlabel('mh1 [GeV]')
                plt.ylabel('mh2 [GeV]')
                plt.title('20'+year)
                plt.savefig(plotdirbase+"ratio_GP_2d.png")
                plt.savefig(plotdirbase+"ratio_GP_2d.pdf")
                plt.clf()
                plt.close()

                hist_ratio_2 = hist_temp / (pred_res_temp + hist_resonant_temp)
                plt.figure(figsize=(6, 5))
                mesh = plt.pcolormesh(xedges, yedges, hist_ratio_2.T, cmap='viridis', edgecolors='none')
                plt.grid(False)
                plt.colorbar(mesh, label='Data / '+process+' + GP (Fit on Data - '+process+')')
                plt.xlabel('mh1 [GeV]')
                plt.ylabel('mh2 [GeV]')
                plt.title('20'+year)
                plt.savefig(plotdirbase+"ratio_GP_res_2d.png")
                plt.savefig(plotdirbase+"ratio_GP_res_2d.pdf")
                plt.clf()
                plt.close()



        #get GP/GP ratio in full massplane

        pred, cov = gpr.predict(gridpoints_scaled, return_cov = True)
        pred_res, cov_res = gpr_res.predict(gridpoints_scaled, return_cov = True)

        if anscombe:

            pred, cov = scale_up_hist(pred, cov, shift, scale)
            pred, cov = undo_anscombe(pred, cov)

            pred_res, cov_res = scale_up_hist(pred_res, cov_res, shift_res, scale_res)
            pred_res, cov_res = undo_anscombe(pred_res, cov_res)
                
        pred_temp = pred.reshape(nbins,nbins)
        pred_res_temp = pred_res.reshape(nbins,nbins)
        hist_resonant_temp = hist_resonant.reshape(nbins,nbins)

        hist_ratio_2 = pred_temp / (pred_res_temp + hist_resonant_temp)
        plt.figure(figsize=(6, 5))
        mesh = plt.pcolormesh(xedges, yedges, hist_ratio_2.T, cmap='viridis', edgecolors='none')
        plt.grid(False)
        plt.colorbar(mesh, label='GP / '+process+' + GP (Fit on Data - '+process+')')
        plt.xlabel('mh1 [GeV]')
        plt.ylabel('mh2 [GeV]')
        plt.title('20'+year)
        plt.savefig(plotdirbase+"ratio_GP_GP_res_2d.png")
        plt.savefig(plotdirbase+"ratio_GP_GP_res_2d.pdf")
        plt.clf()
        plt.close()