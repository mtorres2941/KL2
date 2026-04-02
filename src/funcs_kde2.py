################################################################################################################
################################################################################################################
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm
from customstats import weighted_bw

# edited on 2026-04-01

################################################################################################################
################################################################################################################


def find_area_iqr(xplot, yplot, lo=0.25, hi=0.75):
    """
    INPUT
        xplot    1-d array with x values
        yplot    2-d array with multiple sets of y-values. returned from kl2()
        lo       lower quantile
        hi       upper quantile
    
    OUTPUT
        area_iqr   array between upper and lower quantiles for the plot. Higher metric means greater uncertainty about the probabilistic model
    """
    qlo = np.quantile(yplot, lo, axis=1)
    qhi = np.quantile(yplot, hi, axis=1)
    return np.trapz(qhi, xplot) - np.trapz(qlo, xplot)

################################################################################################################
################################################################################################################

def check_array(array, data, label):
    """
    INPUT: array, data, label
    OUTPUT: ValueError if array and data are different shapes, otherwise array as flattened array
    """
    array = np.array(array).flatten()
    a = array.shape
    b = data.shape
    if a != b:
        raise ValueError(f'{label} shape {a} must be the same as data shape {b}')
    else:
        return array.astype(float)






################################################################################################################
################################################################################################################

def kl2_plot(xplot, yplot, ax, lo=0.25, hi=0.75, color='tab:red', include_area_iqr=True):
    """
    INPUTS: xplot, yplot (output from kl2), ax, lo=0.1, hi=0.9
    OUTPUTS: plot, no return
    """
    loiqr = r'$\mathrm{\mathsf{   _{IQR}   }}$'
    qlo = np.quantile(yplot, lo, axis=1)
    mean = np.mean(yplot.T, axis=0)
    mean /= scipy.integrate.trapezoid(mean, xplot)
    qhi = np.quantile(yplot, hi, axis=1)
    nruns = yplot.shape[0]

    # fig, ax = plt.subplots(1,1)
    #plot regular kde applied to data
    # sns.kdeplot(data=data, bw_method='silverman', label='Regular KDE', ax=ax)

    # plot iterations
    ax.plot(xplot, yplot, alpha=0.01, color='black');
    ax.plot(xplot, mean, color='gray', label=f"{yplot.shape[1]} iterations");

    # plot quartiles
    ax.plot(xplot, mean, color=color, label='Mean PDF');
    ax.plot(xplot, qlo, color=color, linestyle='--',
            label=f"{int(np.round(lo*100,0))}-{int(np.round(hi*100,0))}%");
    ax.plot(xplot, qhi, color=color, linestyle='--');
    ax.fill_between(xplot, qlo, qhi, color=color, alpha=0.5, label=f'A{loiqr}')

    #uncertainty metric
    area_iqr = find_area_iqr(xplot, yplot, lo=lo, hi=hi)
    if include_area_iqr:
        ax.text(0.95, 0.95, f"A{loiqr}={'%.2f' % np.round(area_iqr,2)}", ha='right', va='top', transform=ax.transAxes)

    # format
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1,0.75));
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_ylim(0,)
    # ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black');
    # ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color='black');
   



    ################################################################################################################
################################################################################################################

def kl2(data,
        xplot,
        group_constraints=None,
        BW_factors=None, 
        evtargets=None,
        represented=None,
        nruns=1000,
        progressbar=True
       ):
    # group_constraints=None
    # BW_factors=None
    # evtargets=None
    # represented=None
    # nruns=1000
    # progressbar=True
    """
    This function performs advanced KDE to a dataset, accommodating 
    multiple sources of uncertainty and conveying a series of possible 
    distributions as part of a Dirichlet Process.

    INPUTS:
        data                All data points considered for this analysis in array-like format
        xplot               This must be a user input so you know how to plot the resulting yplot. Use np.linspace(start, stop, num)
        group_constraints   This is a dictionary where the key is a tuple of indices and the value is the proportion of weight those indices should take up
        BW_factors          Bandwidth factors for each point in "data". Mean will be taken as one
        evtargets           The set of feasible targets for expected value of the resulting plot. These values are used to create a KDE plot, which is sampled from for each iteration's target expected value
        represented         What percentage of the overall data is represented by this dataset. Weight represented by new, random values
        nruns               How many probability density functions are produced. Recommended minimum 1000, which is the autofill value
        progressbar         Boolean that dictates whether messages are printed or not. This should be True unless you're running many separate simulations in a row.

    OUTPUTS:
        xplot               Array of x values produced with np.linspace(). May have been resized to accommodate high values
        yplot               Array of (xplot * nruns) representing all resulting probability density functions
        dict_analytics      Dictionary of some results, including all_x, all_w, and all_ev
    """
    ####################################################################################
    ######################  CHECK INPUT DATA ###########################################
    ####################################################################################
    X = np.array(data).flatten()
    lst1 = []
    lst2 = []
    messages = []
    #check group_constraints and represented
    if group_constraints is None:
        if represented is None:
            represented = 0.6
            messages.append('represented has defaulted to 0.6, indicating that this dataset represents 60% of all values')
        elif represented > 1 or represented < 0:
            raise ValueError("The variable 'represented' must be between 0 and 1")        
        group_constraints = {
            tuple(np.arange(len(X))): represented,
        }
        messages.append('No group constraints were provided')
    else:
        # check for repeating indices
        gc_inds = [item for lst in group_constraints.keys() for item in lst]
        repeats = set([x for x in gc_inds if gc_inds.count(x) > 1])
        if len(repeats)>0:
            raise ValueError(f'The following indices in group_constraints are repeating: {repeats}')

        # check for out of bounds indices
        oob = [ele for ele in gc_inds if ele>=len(data)]
        if len(oob)>0:
            raise ValueError(f'group_constraints indices out of bounds. Dataset length: {len(X)}. Indices: {oob}')

        # check for unused indices
        ind_remainder = [ele for ele in np.arange(len(X)) if ele not in gc_inds]
        gc_sum = sum(group_constraints.values())

        # check if represented and group_constraints agree - allocate weight as needed.
        if len(ind_remainder) > 0: #there are remaining indices
            if represented is None:
                raise ValueError(f'Indices were left out of group_constraints and represented was not given. One of these must be corrected. Indices include {ind_remainder}')
            elif represented > 1 or represented < 0:
                raise ValueError("The variable 'represented' must be between 0 and 1")    
            elif represented < gc_sum:
                raise ValueError(f'represented is less than the sum of group_constraints, so there is no remaining weight to be allocated to unconstrained indices: {ind_remainder}')
            group_constraints[tuple(ind_remainder)] = represented-gc_sum

        else: #there are no more remaining indices
            if represented is None:
                represented = gc_sum
                messages.append(f'represented defaulted to the sum of group_constraints: {gc_sum}')
            elif represented > 1 or represented < 0:
                raise ValueError("The variable 'represented' must be between 0 and 1")    
            elif represented != gc_sum:
                messages.append(f'represented does not equal the sum of group_constraints and all indices are accounted for, so represented is being changed from {represented} to {gc_sum}')
                represented = gc_sum
    
    gc_sum = sum(group_constraints.values())
    if represented != gc_sum:
        raise ValueError(f'represented does not equal the sum of group constraints. represented={represented}, sum(gc)={gc_sum}')
    elif represented <= 0 or represented > 1.0:
        raise ValueError(f'represented must be greater than zero and less than or equal to one. Input was {represented}')
        
    # get average weight
    Wavg = np.zeros_like(X).astype(float)
    for inds, perc in group_constraints.items():
        for ind in inds:
            Wavg[ind] = perc/len(inds)

    #check BW_factors
    wstd = Wavg/sum(Wavg)
    base_ev = sum(X*wstd)/sum(wstd)
    # base_std = np.sqrt(sum(wstd*(X-sum(X*wstd)/sum(wstd))**2)/sum(wstd))    
    # iqr = weighted_quantile(X, wstd, 0.75, output='perc2val') - weighted_quantile(X, wstd, 0.25, output='perc2val')
    # bw_base = 0.9 * min([base_std, iqr/1.34]) * len(X)**-0.2
    bw_base = weighted_bw(X, wstd, bw_method='silverman')
    if BW_factors is None:
        # BW_factors = np.ones(len(X)+1)
        BW_factors = np.ones(len(X))
        messages.append('BW_factors has defaulted an array of ones so all bandwidths will be equal.')

    else:
        BW_factors = check_array(BW_factors, X, 'BW_factors')
        BW_factors = BW_factors / np.mean(BW_factors)
        # BW_factors = np.concatenate([BW_factors, [1]])

    #check evtargets
    if evtargets is None:
        evtargets = np.random.normal(base_ev, bw_base, nruns)
        messages.append(f'evtargets has defaulted to a normal distribution with mean = expected value, stdev = bandwidth calculated with the Silverman method')
    else:
        evtargets = np.array(evtargets).flatten()        

    #check xplot - must be custom input array, likely from np.linspace(low, high, number_of_steps)
    if len(xplot) == nruns:
        nruns += 1
        messages.append('nruns has been increased by 1 to avoid being the same integer as xplot. This is to avoid confusing dimensions of the results.')


    if progressbar:
        for message in messages:
            print(message)
            
            
    ####################################################################################
    ######################  FUNCTION STARTS HERE #######################################
    ####################################################################################

    #collect metadata
    n_added = 1
    W_all = np.zeros(shape=(nruns,len(X)+n_added))
    X_all = np.zeros(shape=(nruns,len(X)+n_added))
    EV_all = np.zeros_like(range(nruns))
    yplot = np.zeros_like(xplot, shape=(nruns,len(xplot)))
    xplot_bust = []
        
    #run Monte Carlo simulations
    for run in tqdm(range(nruns), desc='FOR LOOP PROGRESS', disable=not progressbar):
        # reset data
        X = np.array(data).flatten()
        Wavg = Wavg[:len(X)]
        BW_factors = BW_factors[:len(X)]
        
        # pull target
        target = evtargets[run % len(evtargets)]

        # randomly sample weights from Dirichlet distribution
        W = np.zeros_like(X).astype(float)
        for inds, perc in group_constraints.items():
            wnew = np.random.dirichlet(np.ones(len(inds)))*perc
            for ind, w in zip(inds, wnew):
                W[ind] = w.copy()

        # simulate phantom kernel
        EV = sum(X*W)/sum(W)
        if represented == 1:
            xnew = np.array(target).flatten()
            wnew = np.array(0).flatten()
        else:
            xnew = target + (target-EV)*(represented)/(1-represented)
            xnew = np.array(np.max([xnew, 0])).flatten()
            wnew = np.array(1-represented).flatten()

        # calculate bandwidth such that variance of the resulting plot >= variance of the original plot
        # without phantom kernel
        # BW1 = bw_dirichlet(X, Wavg, BW_factors, Wrun=W, bw_method='silverman')
        # BW1 = np.concatenate([BW1, [np.mean(BW1)]])
        # # with phantom kernel
        # X = np.concatenate([np.array(X).flatten(), xnew])
        # Wavg = np.concatenate([Wavg, [1-represented]])
        # BW_factors = np.concatenate([BW_factors, [1.]])
        # W = np.concatenate([np.array(W).flatten(), wnew])
        # BW2 = bw_dirichlet(X, Wavg, BW_factors, Wrun=W, bw_method='silverman')
        # BW = np.max([BW1, BW2], axis=0)
        
        # calculate bandwidth with baseline bandwidth
        BW_factors = np.concatenate([BW_factors, [1.]])
        BW = BW_factors*bw_base
        W = np.concatenate([np.array(W).flatten(), wnew])
        X = np.concatenate([np.array(X).flatten(), xnew])


        # re calculate bandwidth each time
        # bw_basenew = weighted_bw(X, W, bw_method='silverman')
        # BW = BW_factors*bw_basenew
        # W = np.concatenate([np.array(W).flatten(), wnew])
        # X = np.concatenate([np.array(X).flatten(), xnew])

        
        # record results        
        W_all[run] = W
        X_all[run] = X
        EV_all[run] = sum(X*W)/sum(W)

        #check to see if xplot captures everything
        hi = np.max(X + 3*BW)
        if max(xplot) < hi:
            xplot_bust.append(hi)
            # xplot = np.linspace(0,hi,len(xplot))
            # if progressbar:
                # print(f'The upper limit of xplot has been increased to accommodate the upper limit: {hi}')
                # print(f'Consider adjusting the upper limit of xplot. np.max(X + 3*BW) is {np.max(X + 3*BW)} and np.max(xplot) is {np.max(xplot)}')
        
        # create each PDF and ensure it's valid and sums to 1
        yi = norm.pdf(xplot.reshape(-1,1), X, BW)*W
        # yplot[run] = sum(yi.T)
        yplot[run] = yi.sum(axis=1)
        yplot[run] /= np.trapz(yplot[run], xplot)

    dict_analytics = {}
    dict_analytics['all_x'] = X_all
    dict_analytics['all_w'] = W_all
    dict_analytics['all_ev'] = EV_all
    if len(lst1) > 0:
        dict_analytics['lst1'] = lst1
    if len(lst2) > 0:
        dict_analytics['lst2'] = lst2
    if len(xplot_bust) > 0 and progressbar:
        print(f'X+3*BW was greater than max(xplot)={np.max(xplot)} for {len(xplot_bust)}/{nruns} runs, with a max of {np.max(xplot_bust)}')


    return xplot, yplot.T, dict_analytics

