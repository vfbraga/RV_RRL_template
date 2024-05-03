import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize
from mpfit import mpfit
import os
# from astrolibpy import mpfit

def weighted_avg_and_std(values, weights):
    """
    Returns the weighted average and standard deviation.

    :param values: values for which the weighted average should be computed (numpy.ndarray)
    :param weights: weights relative to values. (numpy.ndarray)
    :return y: weighted average and variance (set)
    """
    values=np.asarray(values)
    weights=np.asarray(weights)
    
    average = np.average(values, weights=weights)
    sumw = np.sum(weights)
    
    rescale_coeff = 1.
    if len(values) > 1:
        rescale_coeff = len(values)/(len(values)-1)
    
#     variance = np.average((values-average)**2, weights=weights)
    variance_bevington = rescale_coeff*np.sum(weights*(values-average)**2)/sumw
    
    err = np.sqrt(1/sumw)
    err_variance = np.sqrt(variance_bevington + err**2)
    
    return (average, err_variance, err)

def load_coefficient_table(filein):
    """
    Returns the a pandas.DataFrame object containing the coefficients of the 
    analytical forms of the templates.

    :param filein: string with the entire path to the coefficients.csv file (str).
    :return df_coeff: table of the coefficients of the templates (pandas.DataFrame)
    """

    df_coeff = pd.read_csv(filein, delim_whitespace=True)
    return df_coeff

def gaupe(x, c):

    '''
    gaupe function calculates the value of a Pegasus function up to the
    8th order

    Call: y = RV_template_functions.gaupe(phase,coeff)

    :param x: phases at which the Pegasus function should be calculated (array or list)
    :param c: coefficients of the Pegasus function to be adopted. (pandas.dataFrame)
    :return y: values of the Pegasus function, with c coefficients, at phases x (list)
    '''

    y = c.ZP.values+ \
        c.Amp1.values*np.exp(-(np.sin(np.pi*(x-c.phi1.values))/(c.sig1.values))**2)+ \
        c.Amp2.values*np.exp(-(np.sin(np.pi*(x-c.phi2.values))/(c.sig2.values))**2)+ \
        c.Amp3.values*np.exp(-(np.sin(np.pi*(x-c.phi3.values))/(c.sig3.values))**2)+ \
        c.Amp4.values*np.exp(-(np.sin(np.pi*(x-c.phi4.values))/(c.sig4.values))**2)+ \
        c.Amp5.values*np.exp(-(np.sin(np.pi*(x-c.phi5.values))/(c.sig5.values))**2)+ \
        c.Amp6.values*np.exp(-(np.sin(np.pi*(x-c.phi6.values))/(c.sig6.values))**2)+ \
        c.Amp7.values*np.exp(-(np.sin(np.pi*(x-c.phi7.values))/(c.sig7.values))**2)+ \
        c.Amp8.values*np.exp(-(np.sin(np.pi*(x-c.phi8.values))/(c.sig8.values))**2)
    return y

def gaupe_for_templfit_amplfixed(x, Delta_phase, Delta_mag, ARV,
                          templatebin_int, diagnostic_int, filein='coefficients.csv'):

    '''
    gaupe function calculates the value of a Pegasus function up to the
    8th order

    :param x: phases at which the Pegasus function should be calculated (array or list)
    :param c: [Deltamag, Deltaphase, ARV, bin, diagnostic] . (list)
    :return y: values of the Pegasus function, with c coefficients, at phases x (list)
    '''

    templatebin_dict = {0:'RRc', 1:'RRab1', 2:'RRab2', 3:'RRab3'}
    diagnostic_dict = {0:'Fe', 1:'Na', 2:'Mg', 3:'Ha', 4:'Hb', 5:'Hg', 6:'Hd'}

    templatebin = templatebin_dict[templatebin_int]
    diagnostic = diagnostic_dict[diagnostic_int]

    # Read the coefficients table
    coeff = load_coefficient_table(filein)

    # select the right template for the current diagnostic and template bin
    coeff = coeff[(coeff['Template'] == diagnostic) & (coeff['Bin'] == templatebin)]

    # y = gaupe(x+c[1], coeff)
    # y = c[0] + y * c[2]

    Delta_phase = float(Delta_phase)
    Delta_mag = float(Delta_mag)
    ARV = float(ARV)

    y = gaupe(x+Delta_phase, coeff)
    y = Delta_mag + y * ARV

    return y


def myfunct_gaupe_for_templfit_amplfixed(p, fjac=None, x=None, y=None, err=None):
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    model = gaupe_for_templfit_amplfixed(x, p[0], p[1], p[2], p[3], p[4])
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    return [status, (y-model)/err]


def correct_phase(phase, pulsation_type, diagnostic_int, period, tmean_or_tmax):

    '''
    Corrects the phase in the case that it was derived by using Tmax and/or
    if Balmer lines templates have to be used.

    :param phase: input phase (list)
    :param pulsation_type: pulsation_type (int)
     0 for Fundamental
     1 for First Overtone
    :param diagnostic_int: chemical element that was used to measure RVs.
     Possible values: 0, 1, 2, 3, 4, 5, 6 for
     Iron, Magnesium, Sodium, H_alpha, H_beta, H_gamma, H_delta, respectively
    :param period: pulsation period in days (np.float64)
    :param tmean_or_tmax: Possible values: 'Tmean'; 'Tmax'. Indicate whether the available
     anchor epoch is the epoch of maximum or the epoch of the mean
     magnitude on the rising branch.
    :return: phase
    '''

    correction_for_tmax = False
    # Correct the phase if Tmax instead of Tmean is provided
    if tmean_or_tmax == 'Tmax':
        correction_for_tmax = True
        if pulsation_type == 0:
            phase = phase + 0.223
        else:
            phase = phase + 0.043 + 0.099 * period

    correction_for_hb = False
    # Correct the phase if the template has to be applied to Balmer lines
    if diagnostic_int in [3, 4, 5, 6]:
        correction_for_hb = True
        phase = phase + 0.023 - 0.096 * period

    if correction_for_tmax | correction_for_hb:
        phase = (phase + 1) % 1

    return phase

def amplitude_rescale(AV, pulsation_type, diagnostic_int):

    '''
    Function that rescales the V-band amplitude into radial velocity amplitude

    :param AV: V-band amplitude, in magnitudes, of the target (float)
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param diagnostic_int: chemical element that was used to measure RVs.
     Possible values: 0, 1, 2, 3, 4, 5, 6 for
     Iron, Magnesium, Sodium, H_alpha, H_beta, H_gamma, H_delta, respectively
    :return ARV: the radial velocity amplitude for the selected diagnostic and pulsation  mode
    '''

    if diagnostic_int == 3: # Halpha
        if pulsation_type == 0:
            ARV = 77.77 + 22.83 * AV
        else:
            ARV = -9.18 + 106.69 * AV
    elif diagnostic_int == 4: # Hbeta
        if pulsation_type == 0:
            ARV = 63.32 + 15.96 * AV
        else:
            ARV = -0.27 + 66.55 * AV
    elif diagnostic_int == 5: # Hgamma
        if pulsation_type == 0:
            ARV = 57.38 + 18.48 * AV
        else:
            ARV = 0.54 + 59.32 * AV
    elif diagnostic_int == 6: # Hdelta
        if pulsation_type == 0:
            ARV = 50.90 + 14.43 * AV
        else:
            ARV = 3.85 + 45.78 * AV
    else:
        if pulsation_type == 0: # Fe Mg Na
            ARV = 38.09 + 22.35 * AV
        else:
            ARV = 1.68 + 52.63 * AV

    return ARV

def find_templatebin(pulsation_type,period):

    '''
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param period: pulsation period in days (np.float64)
    :return templatebin: index to search in the coefficient table (string)
    '''

    if pulsation_type == 1:
        templatebin = 0
    elif pulsation_type == 0:
        if period < 0.55:
            templatebin = 1
        elif period > 0.7:
            templatebin = 3
        else:
            templatebin = 2

    return(templatebin)

def apply_template_anchor(HJD, RV, errRV, AV, pulsation_type,
                            period, t0, tmean_or_tmax, diagnostic_int,
                            filein, figure_out=''):

    '''
    apply_template_anchor function applies the right template (selected
    by means of the parameters pulsation_type, period and diagnostic)
    on a series of RV measurements

    :param HJD: list of Heliocentric Julian Dates for the RV measurements (list)
    :param RV: list of RV measurements (list)
    :param errRV: list of uncertainties on RV (list)
    :param AV: V-band amplitude, in magnitudes, of the target (float)
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param period: pulsation period in days (np.float64)
    :param t0: Anchor epoch in HJD (np.float64)
    :param tmean_or_tmax: Possible values: 'Tmean'; 'Tmax'. Indicate whether the available
     anchor epoch is the epoch of maximum or the epoch of the mean
     magnitude on the rising branch.
    :param diagnostic_int: chemical element that was used to measure RVs.
     Possible values: 0, 1, 2, 3, 4, 5, 6 for
     Iron, Magnesium, Sodium, H_alpha, H_beta, H_gamma, H_delta, respectively
    :param filein: path to the coefficients table in csv format (string)
    :param figure_out: path to the output figure. '' if no output figure is desired. (string)
    :return: data_return: dictionary including the following entries:
     'v_gamma_list': list of systemic velocities from each RV measurement
     'xfit': grid of phases
     'yfit_list': template fit values for each RV measuremnet (list of lists)
     'v_gamma_mean': 2-element tuple including average and standard deviation of the systemic velocity
     (dictionary)
    '''

    templatebin_dict = {0:'RRc', 1:'RRab1', 2:'RRab2', 3:'RRab3'}
    diagnostic_dict = {0:'Fe', 1:'Na', 2:'Mg', 3:'Ha', 4:'Hb', 5:'Hg', 6:'Hd'}

    templatebin_int = find_templatebin(pulsation_type, period)

    templatebin = templatebin_dict[templatebin_int]
    diagnostic = diagnostic_dict[diagnostic_int]

    # Derive the phase from HJD, t0 and period
    phase = (HJD-t0)/period % 1.
    phase = correct_phase(phase, pulsation_type, diagnostic, period, tmean_or_tmax)

    ARV = amplitude_rescale(AV, pulsation_type, diagnostic_int)

    # Generates the grid of phases to evaluate the template
    n_phases_for_model = 1000
    xfit = (np.arange(n_phases_for_model + 1) / float(n_phases_for_model))

    # Read the coefficients table
    c = load_coefficient_table(filein)

    # select the right template for the current diagnostic and template bin
    c = c[(c['Template'] == diagnostic) & (c['Bin'] == templatebin)]
    
    # Estimate V_gamma for each RV measurement
    v_gamma_list = []
#     err_v_gamma_list = []
    yfit_list = []

    for phase_i, RV_i, errRV_i in zip(phase, RV, errRV):

        # Calculates the value of the template at the phase of the observed RV measurement
        template_value_at_phase = gaupe(phase_i, c)

        # Calculates the systemic velocity
        v_gamma_temp = RV_i - template_value_at_phase * ARV

        # Calculates the RV curve model anchored to the RV measurement
        yfit = v_gamma_temp + gaupe(xfit, c) * ARV

        v_gamma_list.append(v_gamma_temp[0])
#         err_v_gamma_list.append(errRV_i)
        yfit_list.append(yfit)

    v_gamma_list = np.asarray(v_gamma_list)
    v_gamma_mean = weighted_avg_and_std(v_gamma_list, 1./(errRV*errRV))
    
    data_return = {'v_gamma_list': v_gamma_list, 'xfit': xfit, 'yfit_list': yfit_list,
                   'v_gamma_mean': v_gamma_mean[0], 
                   'errv_gamma_mean': np.sqrt(v_gamma_mean[1]**2 + (ARV*c.sigma.values[0])**2)}

    if figure_out != '':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        cmap_size = 256
        cmap = cm.get_cmap('Oranges', cmap_size)
        cmap_step = np.ceil(cmap_size/len(phase))
        cmap_zero = np.floor(cmap_step/2)

        iii = 0
        for phase_i, RV_i, errRV_i in zip(phase, RV, errRV):

            ax.plot(xfit, yfit_list[iii], c=cmap(int(cmap_zero + iii*cmap_step)), zorder=0)
            ax.plot([-1, 2], [v_gamma_list[iii], v_gamma_list[iii]], '--',
                    c=cmap(int(cmap_zero + iii*cmap_step)), zorder=0)
            ax.errorbar(phase_i, RV_i, yerr=errRV_i, c=cmap(int(cmap_zero + iii*cmap_step)), zorder=1)
            ax.scatter(phase_i, RV_i, c=[cmap(int(cmap_zero + iii*cmap_step))], s=20, zorder=1)
            iii = iii + 1

        ax.plot([-1,2], [v_gamma_mean[0], v_gamma_mean[0]], c='k', zorder=2)
        ax.set_xlim([0.,1.])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('PHASE', fontsize=18)
        plt.ylabel('RV [km/s]', fontsize=18)
        fig.savefig(figure_out)
        plt.close()

    return data_return


def apply_template_templfit_amplfixed(HJD, RV, errRV, AV, pulsation_type,
                            period, t0, diagnostic_int,
                            filein, figure_out='', quiet=1):

    '''
    apply_template_templfit_amplfixed function applies the right template (selected
    by means of the parameters pulsation_type, period and diagnostic)
    on a series of RV measurements

    :param HJD: list of Heliocentric Julian Dates for the RV measurements (list)
    :param RV: list of RV measurements (list)
    :param errRV: list of uncertainties on RV (list)
    :param AV: V-band amplitude, in magnitudes, of the target (float)
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param period: pulsation period in days (np.float64)
    :param t0: Anchor epoch in HJD (np.float64)
    :param diagnostic_int: chemical element that was used to measure RVs.
     Possible values: 0, 1, 2, 3, 4, 5, 6 for
     Iron, Magnesium, Sodium, H_alpha, H_beta, H_gamma, H_delta, respectively
    :param filein: path to the coefficients table in csv format (string)
    :param figure_out: path to the output figure. '' if no output figure is desired. (string)
    :param quiet: 0/1 to allow/forbid mpfit to print the results of the iterations (int)
    :return: data_return: dictionary including the following entries:
     'v_gamma_list': list of systemic velocities from each RV measurement
     'xfit': grid of phases
     'yfit_list': template fit values for each RV measuremnet (list of lists)
     'v_gamma_mean': 2-element tuple including average and standard deviation of the systemic velocity
     (dictionary)
    '''

    templatebin_dict = {0:'RRc', 1:'RRab1', 2:'RRab2', 3:'RRab3'}
    diagnostic_dict = {0:'Fe', 1:'Na', 2:'Mg', 3:'Ha', 4:'Hb', 5:'Hg', 6:'Hd'}

    templatebin_int = find_templatebin(pulsation_type, period)

    templatebin = templatebin_dict[templatebin_int]
    diagnostic = diagnostic_dict[diagnostic_int]

    # Derive the phase from HJD, t0 and period
    phase = (HJD-t0)/period % 1.

    ARV = amplitude_rescale(AV, pulsation_type, diagnostic_int)

    # Generates the grid of phases to evaluate the template
    n_phases_for_model = 1000
    xfit = (np.arange(n_phases_for_model + 1) / float(n_phases_for_model))

    # Read the coefficients table
    c = load_coefficient_table(filein)

    # select the right template for the current diagnostic and template bin
    c = c[(c['Template'] == diagnostic) & (c['Bin'] == templatebin)]

    # Generates the first guess on the coefficients
    n_guesses = 3
    deltaphase_guesses = np.arange(n_guesses)/float(n_guesses)

    chisqs=[]
    popts=[]
    for deltaphase_guess in deltaphase_guesses:

        p0 = (deltaphase_guess, np.mean(RV), ARV, templatebin_int, diagnostic_int)

        parinfo = [{'value': deltaphase_guess, 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0]},
                   {'value': np.mean(RV), 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0]},
                   {'value': ARV, 'fixed': 1},
                   {'value': templatebin_int, 'fixed': 1},
                   {'value': diagnostic_int, 'fixed': 1}]

        fa = {'x': phase, 'y': RV, 'err': errRV}
        m = mpfit(myfunct_gaupe_for_templfit_amplfixed, p0, parinfo=parinfo, functkw=fa, quiet=quiet)

        # yfit = gaupe_for_templfit_amplfixed(xfit, *m.params, filein)
        chisq = (myfunct_gaupe_for_templfit_amplfixed(m.params, x=phase, y=RV, err=errRV)[1] ** 2).sum() / (len(RV)-2)
        chisqs.append(chisq)
        popts.append(m.params)

    chisqs = np.asarray(chisqs)
    ind_best = chisqs.argmin()

    yfit = gaupe_for_templfit_amplfixed(xfit, *popts[ind_best], filein)
    v_gamma_mean = np.mean(yfit)
    errv_gamma_mean = np.sqrt(np.diag(m.covar))[1]
#     errv_gamma_mean = 0

    if figure_out != '':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

# to plot all the attempts with different phase_guesses
       #  colors=['r','b','g']
       # for iii, popt in enumerate(popts):
       #      print('---')
       #      print(popt)
       #      yfit = gaupe_for_templfit_amplfixed(xfit, *popt, filein)
       #      # yfit = gaupe_for_templfit_amplfixed(xfit, popt[0], popt[1], ARV, templatebin, diagnostic, filein)
       #
       #      ax.plot(xfit, yfit, colors[iii]+'--', zorder=0)
       #      ax.text(xfit[0], yfit[0], str(iii))

        ax.plot(xfit, yfit, 'k', zorder=0)

        ax.plot([-1, 2], [v_gamma_mean, v_gamma_mean], '--', c='k', zorder=0)
        ax.errorbar(phase, RV, yerr=errRV, c='r', fmt = ' ', zorder=1)
        ax.scatter(phase, RV, c='r', s=20, zorder=1)

        ax.set_xlim([0.,1.])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('PHASE', fontsize=18)
        plt.ylabel('RV [km/s]', fontsize=18)
        fig.savefig(figure_out)
        plt.close()

    return {'xfit': xfit, 'yfit': yfit,
            'v_gamma_mean': v_gamma_mean, 
            'errv_gamma_mean': np.sqrt(errv_gamma_mean**2 + (ARV*c.sigma.values[0])**2), 
            'popts': popts[ind_best],
            'chisq': chisqs[ind_best]}

def apply_template_templfit_amplfree(HJD, RV, errRV, pulsation_type,
                            period, t0, diagnostic_int,
                            filein, figure_out='', quiet=1):

    '''
    apply_template_templfit_amplfixed function applies the right template (selected
    by means of the parameters pulsation_type, period and diagnostic)
    on a series of RV measurements

    :param HJD: list of Heliocentric Julian Dates for the RV measurements (list)
    :param RV: list of RV measurements (list)
    :param errRV: list of uncertainties on RV (list)
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param period: pulsation period in days (np.float64)
    :param t0: Anchor epoch in HJD (np.float64)
    :param diagnostic_int: chemical element that was used to measure RVs.
     Possible values: 0, 1, 2, 3, 4, 5, 6 for
     Iron, Magnesium, Sodium, H_alpha, H_beta, H_gamma, H_delta, respectively
    :param filein: path to the coefficients table in csv format (string)
    :param figure_out: path to the output figure. '' if no output figure is desired. (string)
    :param quiet: 0/1 to allow/forbid mpfit to print the results of the iterations (int)
    :return: data_return: dictionary including the following entries:
     'v_gamma_list': list of systemic velocities from each RV measurement
     'xfit': grid of phases
     'yfit_list': template fit values for each RV measuremnet (list of lists)
     'v_gamma_mean': 2-element tuple including average and standard deviation of the systemic velocity
     (dictionary)
    '''

    templatebin_dict = {0:'RRc', 1:'RRab1', 2:'RRab2', 3:'RRab3'}
    diagnostic_dict = {0:'Fe', 1:'Na', 2:'Mg', 3:'Ha', 4:'Hb', 5:'Hg', 6:'Hd'}

    templatebin_int = find_templatebin(pulsation_type, period)

    templatebin = templatebin_dict[templatebin_int]
    diagnostic = diagnostic_dict[diagnostic_int]

    # Derive the phase from HJD, t0 and period
    phase = (HJD-t0)/period % 1.

    # Generates the grid of phases to evaluate the template
    n_phases_for_model = 1000
    xfit = (np.arange(n_phases_for_model + 1) / float(n_phases_for_model))

    # Read the coefficients table
    c = load_coefficient_table(filein)

    # select the right template for the current diagnostic and template bin
    c = c[(c['Template'] == diagnostic) & (c['Bin'] == templatebin)]

    # Generates the first guess on the coefficients
    n_guesses = 3
    deltaphase_guesses = np.arange(n_guesses)/float(n_guesses)

    chisqs=[]
    popts=[]
    for deltaphase_guess in deltaphase_guesses:

        p0 = (deltaphase_guess, np.mean(RV), np.max(RV) - np.min(RV), templatebin_int, diagnostic_int)

        parinfo = [{'value': deltaphase_guess, 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0]},
                   {'value': np.mean(RV), 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0]},
                   {'value': np.max(RV) - np.min(RV), 'fixed': 0, 'limited': [1, 0], 'limits': [0.0, 0.0]},
                   {'value': templatebin_int, 'fixed': 1},
                   {'value': diagnostic_int, 'fixed': 1}]

        fa = {'x': phase, 'y': RV, 'err': errRV}
        m = mpfit(myfunct_gaupe_for_templfit_amplfixed, p0, parinfo=parinfo, functkw=fa, quiet=quiet)

        # yfit = gaupe_for_templfit_amplfixed(xfit, *m.params, filein)
        chisq = (myfunct_gaupe_for_templfit_amplfixed(m.params, x=phase, y=RV, err=errRV)[1] ** 2).sum() / (len(RV)-2)
        chisqs.append(chisq)
        popts.append(m.params)

    chisqs = np.asarray(chisqs)
    ind_best = chisqs.argmin()    
    
    yfit = gaupe_for_templfit_amplfixed(xfit, *popts[ind_best], filein)
    v_gamma_mean = np.mean(yfit)
    errv_gamma_mean = np.sqrt(np.diag(m.covar))[1]

    if figure_out != '':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        ax.plot(xfit, yfit, 'k', zorder=0)

        ax.plot([-1, 2], [v_gamma_mean, v_gamma_mean], '--', c='k', zorder=0)
        ax.errorbar(phase, RV, yerr=errRV, c='r', fmt = ' ', zorder=1)
        ax.scatter(phase, RV, c='r', s=20, zorder=1)

        ax.set_xlim([0.,1.])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('PHASE', fontsize=18)
        plt.ylabel('RV [km/s]', fontsize=18)
        fig.savefig(figure_out)
        plt.close()

    return {'xfit': xfit, 'yfit': yfit,
            'v_gamma_mean': v_gamma_mean, 
            'errv_gamma_mean': np.sqrt(errv_gamma_mean**2 + ( popts[ind_best][2]*c.sigma.values[0])**2 ), 
            'popts': popts[ind_best],
            'chisq': chisqs[ind_best]}

def apply_template_templfit_amplfixed_fromphases(phase, RV, errRV, AV, pulsation_type,
                            period, diagnostic_int,
                            filein, figure_out='', quiet=1):

    '''
    apply_template_templfit_amplfixed function applies the right template (selected
    by means of the parameters pulsation_type, period and diagnostic)
    on a series of RV measurements. It can be used if only phases and not HJDs are available

    :param HJD: list of Heliocentric Julian Dates for the RV measurements (list)
    :param RV: list of RV measurements (list)
    :param errRV: list of uncertainties on RV (list)
    :param AV: V-band amplitude, in magnitudes, of the target (float)
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param period: pulsation period in days (np.float64)
    :param diagnostic_int: chemical element that was used to measure RVs.
     Possible values: 0, 1, 2, 3, 4, 5, 6 for
     Iron, Magnesium, Sodium, H_alpha, H_beta, H_gamma, H_delta, respectively
    :param filein: path to the coefficients table in csv format (string)
    :param figure_out: path to the output figure. '' if no output figure is desired. (string)
    :param quiet: 0/1 to allow/forbid mpfit to print the results of the iterations (int)
    :return: data_return: dictionary including the following entries:
     'v_gamma_list': list of systemic velocities from each RV measurement
     'xfit': grid of phases
     'yfit_list': template fit values for each RV measuremnet (list of lists)
     'v_gamma_mean': 2-element tuple including average and standard deviation of the systemic velocity
     (dictionary)
    '''

    templatebin_dict = {0:'RRc', 1:'RRab1', 2:'RRab2', 3:'RRab3'}
    diagnostic_dict = {0:'Fe', 1:'Na', 2:'Mg', 3:'Ha', 4:'Hb', 5:'Hg', 6:'Hd'}

    templatebin_int = find_templatebin(pulsation_type, period)

    templatebin = templatebin_dict[templatebin_int]
    diagnostic = diagnostic_dict[diagnostic_int]

    ARV = amplitude_rescale(AV, pulsation_type, diagnostic_int)

    # Generates the grid of phases to evaluate the template
    n_phases_for_model = 1000
    xfit = (np.arange(n_phases_for_model + 1) / float(n_phases_for_model))

    # Read the coefficients table
    c = load_coefficient_table(filein)

    # select the right template for the current diagnostic and template bin
    c = c[(c['Template'] == diagnostic) & (c['Bin'] == templatebin)]

    # Generates the first guess on the coefficients
    n_guesses = 3
    deltaphase_guesses = np.arange(n_guesses)/float(n_guesses)

    chisqs=[]
    popts=[]
    for deltaphase_guess in deltaphase_guesses:

        p0 = (deltaphase_guess, np.mean(RV), ARV, templatebin_int, diagnostic_int)

        parinfo = [{'value': deltaphase_guess, 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0]},
                   {'value': np.mean(RV), 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0]},
                   {'value': ARV, 'fixed': 1},
                   {'value': templatebin_int, 'fixed': 1},
                   {'value': diagnostic_int, 'fixed': 1}]

        fa = {'x': phase, 'y': RV, 'err': errRV}
        m = mpfit(myfunct_gaupe_for_templfit_amplfixed, p0, parinfo=parinfo, functkw=fa, quiet=quiet)

        # yfit = gaupe_for_templfit_amplfixed(xfit, *m.params, filein)
        chisq = (myfunct_gaupe_for_templfit_amplfixed(m.params, x=phase, y=RV, err=errRV)[1] ** 2).sum() / (len(RV)-2)
        chisqs.append(chisq)
        popts.append(m.params)

    chisqs = np.asarray(chisqs)
    ind_best = chisqs.argmin()

    yfit = gaupe_for_templfit_amplfixed(xfit, *popts[ind_best], filein)
    v_gamma_mean = np.mean(yfit)
    errv_gamma_mean = np.sqrt(np.diag(m.covar))[1]
#     errv_gamma_mean = 0

    if figure_out != '':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

# to plot all the attempts with different phase_guesses
       #  colors=['r','b','g']
       # for iii, popt in enumerate(popts):
       #      print('---')
       #      print(popt)
       #      yfit = gaupe_for_templfit_amplfixed(xfit, *popt, filein)
       #      # yfit = gaupe_for_templfit_amplfixed(xfit, popt[0], popt[1], ARV, templatebin, diagnostic, filein)
       #
       #      ax.plot(xfit, yfit, colors[iii]+'--', zorder=0)
       #      ax.text(xfit[0], yfit[0], str(iii))

        ax.plot(xfit, yfit, 'k', zorder=0)

        ax.plot([-1, 2], [v_gamma_mean, v_gamma_mean], '--', c='k', zorder=0)
        ax.errorbar(phase, RV, yerr=errRV, c='r', fmt = ' ', zorder=1)
        ax.scatter(phase, RV, c='r', s=20, zorder=1)

        ax.set_xlim([0.,1.])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('PHASE', fontsize=18)
        plt.ylabel('RV [km/s]', fontsize=18)
        fig.savefig(figure_out)
        plt.close()

    return {'xfit': xfit, 'yfit': yfit,
            'v_gamma_mean': v_gamma_mean, 
            'errv_gamma_mean': np.sqrt(errv_gamma_mean**2 + (ARV*c.sigma.values[0])**2), 
            'popts': popts[ind_best],
            'chisq': chisqs[ind_best]}

def apply_template_templfit_amplfree_fromphases(phase, RV, errRV, pulsation_type,
                            period, diagnostic_int,
                            filein, figure_out='', quiet=1):

    '''
    apply_template_templfit_amplfixed function applies the right template (selected
    by means of the parameters pulsation_type, period and diagnostic)
    on a series of RV measurements. It can be used if only phases and not HJDs are available

    :param phase: list of Heliocentric Julian Dates for the RV measurements (list)
    :param RV: list of RV measurements (list)
    :param errRV: list of uncertainties on RV (list)
    :param pulsation_type: pulsation mode (int)
     0 for Fundamental
     1 for First Overtone
    :param period: pulsation period in days (np.float64)
    :param diagnostic_int: chemical element that was used to measure RVs.
     Possible values: 0, 1, 2, 3, 4, 5, 6 for
     Iron, Magnesium, Sodium, H_alpha, H_beta, H_gamma, H_delta, respectively
    :param filein: path to the coefficients table in csv format (string)
    :param figure_out: path to the output figure. '' if no output figure is desired. (string)
    :param quiet: 0/1 to allow/forbid mpfit to print the results of the iterations (int)
    :return: data_return: dictionary including the following entries:
     'v_gamma_list': list of systemic velocities from each RV measurement
     'xfit': grid of phases
     'yfit_list': template fit values for each RV measuremnet (list of lists)
     'v_gamma_mean': 2-element tuple including average and standard deviation of the systemic velocity
     (dictionary)
    '''

    templatebin_dict = {0:'RRc', 1:'RRab1', 2:'RRab2', 3:'RRab3'}
    diagnostic_dict = {0:'Fe', 1:'Na', 2:'Mg', 3:'Ha', 4:'Hb', 5:'Hg', 6:'Hd'}

    templatebin_int = find_templatebin(pulsation_type, period)

    templatebin = templatebin_dict[templatebin_int]
    diagnostic = diagnostic_dict[diagnostic_int]

    # Generates the grid of phases to evaluate the template
    n_phases_for_model = 1000
    xfit = (np.arange(n_phases_for_model + 1) / float(n_phases_for_model))

    # Read the coefficients table
    c = load_coefficient_table(filein)

    # select the right template for the current diagnostic and template bin
    c = c[(c['Template'] == diagnostic) & (c['Bin'] == templatebin)]

    # Generates the first guess on the coefficients
    n_guesses = 3
    deltaphase_guesses = np.arange(n_guesses)/float(n_guesses)

    os.chdir('/home/vittorioinaf/Documenti/Programmi/Python/RV_RRL_template/')
    
    chisqs=[]
    popts=[]
    for deltaphase_guess in deltaphase_guesses:

        p0 = (deltaphase_guess, np.mean(RV), np.max(RV) - np.min(RV), templatebin_int, diagnostic_int)

        parinfo = [{'value': deltaphase_guess, 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0]},
                   {'value': np.mean(RV), 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0]},
                   {'value': np.max(RV) - np.min(RV), 'fixed': 0, 'limited': [1, 0], 'limits': [0.0, 0.0]},
                   {'value': templatebin_int, 'fixed': 1},
                   {'value': diagnostic_int, 'fixed': 1}]

        fa = {'x': phase, 'y': RV, 'err': errRV}
        m = mpfit(myfunct_gaupe_for_templfit_amplfixed, p0, parinfo=parinfo, functkw=fa, quiet=quiet)

        # yfit = gaupe_for_templfit_amplfixed(xfit, *m.params, filein)
        chisq = (myfunct_gaupe_for_templfit_amplfixed(m.params, x=phase, y=RV, err=errRV)[1] ** 2).sum() / (len(RV)-2)
        chisqs.append(chisq)
        popts.append(m.params)

    chisqs = np.asarray(chisqs)
    ind_best = chisqs.argmin()    
    
    yfit = gaupe_for_templfit_amplfixed(xfit, *popts[ind_best], filein)
    v_gamma_mean = np.mean(yfit)
    errv_gamma_mean = np.sqrt(np.diag(m.covar))[1]

    if figure_out != '':
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        ax.plot(xfit, yfit, 'k', zorder=0)

        ax.plot([-1, 2], [v_gamma_mean, v_gamma_mean], '--', c='k', zorder=0)
        ax.errorbar(phase, RV, yerr=errRV, c='r', fmt = ' ', zorder=1)
        ax.scatter(phase, RV, c='r', s=20, zorder=1)

        ax.set_xlim([0.,1.])
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('PHASE', fontsize=18)
        plt.ylabel('RV [km/s]', fontsize=18)
        fig.savefig(figure_out)
        plt.close()

    return {'xfit': xfit, 'yfit': yfit,
            'v_gamma_mean': v_gamma_mean, 
            'errv_gamma_mean': np.sqrt(errv_gamma_mean**2 + ( popts[ind_best][2]*c.sigma.values[0])**2 ), 
            'popts': popts[ind_best],
            'chisq': chisqs[ind_best]}
