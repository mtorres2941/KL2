
import numpy as np
from scipy.integrate import cumulative_trapezoid
import scipy.interpolate
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from scipy.stats import lognorm, gaussian_kde, wasserstein_distance, energy_distance, norm


# from customstats import effective_variance, weighted_lognorm_fit, shapiro_wilk_weighted, _royston_pvalue, empirical_metadata, NestedDictValues, weighted_ecdf, estimate_maxima, weighted_kurtosis, weighted_skew, wasserstein1_weighted, wasserstein2_weighted, weighted_mean, weighted_var, weighted_distance_norm, weighted_quantile, weighted_bw, weighted_std

########################################################################

def effective_variance(x, pdf):
    norm = np.trapz(pdf, x)
    pdf_norm = pdf / norm
    mu = np.trapz(x * pdf_norm, x)
    return np.trapz(pdf_norm * (x - mu)**2, x)

########################################################################
def weighted_lognorm_fit(data, weights=None, method="MLE"):
    """
    Fit a weighted lognormal distribution to data.
    
    Parameters
    ----------
    data : array-like
        Positive data values.
    weights : array-like, optional
        Non-negative weights for each data point. If None, all weights = 1.
    method : {"MLE", "MoM"}, default="MLE"
        Estimation method:
        - "MLE": weighted maximum likelihood estimation
        - "MoM": method of moments (weighted mean/var in log-space)
    
    Returns
    -------
    shape : float
        Shape parameter (sigma).
    loc : float
        Location parameter (always 0 in this fit).
    scale : float
        Scale parameter (exp(mu)).
    """
    data = np.asarray(data)
    if weights is None:
        weights = np.ones_like(data)
    else:
        weights = np.asarray(weights)
    
    if np.any(data <= 0):
        raise ValueError("Lognormal fit requires strictly positive data")
    
    # Normalize weights
    weights = weights / np.sum(weights)

    # ----- Method of Moments in log-space -----
    if method == "MoM":
        logx = np.log(data)
        mu = np.average(logx, weights=weights)
        var = np.average((logx - mu)**2, weights=weights)
        sigma = np.sqrt(var)
        return sigma, 0.0, np.exp(mu)

    # ----- Maximum Likelihood Estimation -----
    def nll(params):
        sigma, mu = params
        if sigma <= 0:
            return np.inf
        logpdf = lognorm.logpdf(data, s=sigma, scale=np.exp(mu))
        return -np.sum(weights * logpdf)

    # Use weighted log-moments as initialization
    logx = np.log(data)
    mu0 = np.average(logx, weights=weights)
    sigma0 = np.sqrt(np.average((logx - mu0)**2, weights=weights))

    res = minimize(nll, x0=[sigma0, mu0], bounds=[(1e-9, None), (None, None)])
    sigma, mu = res.x
    return sigma, 0.0, np.exp(mu)


# Example usage
# if __name__ == "__main__":
#     rng = np.random.default_rng(0)
#     data = lognorm(s=0.5, scale=np.exp(1.0)).rvs(size=1000, random_state=rng)
#     weights = rng.random(1000)  # arbitrary weights

#     s, loc, scale = weighted_lognorm_fit(data, weights, method="MLE")
#     print(f"Fitted lognorm params: shape={s:.4f}, loc={loc:.4f}, scale={scale:.4f}")

########################################################################
def shapiro_wilk_weighted(x, weights=None):
    """
    Shapiro-Wilk test of normality, extended to handle sample weights.

    For uniformly weighted (or unweighted) data, returns results identical
    to ``scipy.stats.shapiro``. For non-uniform weights the W statistic is
    the squared weighted Pearson correlation between the sorted data and
    their weighted normal scores (a Shapiro-Francia generalisation), and
    the p-value uses the Royston (1992) approximation evaluated at the
    Kish (1965) effective sample size.

    Parameters
    ----------
    x : array-like, shape (n,)
        Sample data. Must contain at least 3 observations.
    weights : array-like, shape (n,), optional
        Non-negative importance/frequency weights. Need not be normalised.
        ``None`` (default) is equivalent to uniform weights and produces
        the same output as ``scipy.stats.shapiro(x)``.

    Returns
    -------
    statistic : float
        The W test statistic (0-1; values near 1 indicate normality).
    pvalue : float
        P-value for the null hypothesis that *x* is normally distributed.

    Notes
    -----
    The weighted W statistic is

        W = Cov_w(x, z)^2 / (Var_w(x) * Var_w(z))

    where z_i = norm.ppf(p_i) and p_i is the midpoint of the i-th step of
    the weighted empirical CDF.  With uniform weights this equals the
    Shapiro-Francia W' (squared correlation with normal scores), which is
    indistinguishable from Shapiro-Wilk W for n >= 20.

    The n > 5000 restriction applies only to the uniform-weight path (where
    scipy enforces it).  For weighted data, only the Kish effective sample
    size n_eff = 1/sum(w_i^2) must be <= 5000, so large raw datasets with
    unequal weights are fully supported.

    The p-value approximation (Royston 1992) is most accurate for n_eff >= 12.

    References
    ----------
    Shapiro & Wilk (1965). Biometrika, 52, 591-611.
    Royston (1992). Statistics and Computing, 2, 117-119.
    Kish (1965). Survey Sampling. Wiley.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if n < 3:
        raise ValueError(f"n must be >= 3, got {n}.")

    # No weights supplied -> exact scipy result (scipy enforces n <= 5000)
    if weights is None:
        return stats.shapiro(x)

    w = np.asarray(weights, dtype=float).ravel()
    if w.shape != (n,):
        raise ValueError("weights must be 1-D and the same length as x.")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    if w.sum() <= 0:
        raise ValueError("Sum of weights must be positive.")

    w = w / w.sum()  # normalise to sum to 1

    # Uniform weights -> exact scipy result
    if np.allclose(w, 1.0 / n):
        return stats.shapiro(x)

    # Kish effective sample size (used for p-value, not the raw n)
    n_eff = max(3, min(5000, int(round(1.0 / (w ** 2).sum()))))

    # Sort
    idx = np.argsort(x)
    x_s = x[idx]
    w_s = w[idx]

    # Weighted ECDF midpoints: p_i = (cumulative weight up to i) - w_i/2
    cum_w = np.cumsum(w_s)
    p = np.clip(cum_w - w_s / 2.0, 1e-10, 1.0 - 1e-10)

    # Normal scores
    z = stats.norm.ppf(p)

    # Weighted W = (weighted correlation between x and z)^2
    x_bar = (w_s * x_s).sum()
    z_bar = (w_s * z).sum()  # ~0 for symmetric weights / large n
    dx = x_s - x_bar
    dz = z  - z_bar
    cov_xz = (w_s * dx * dz).sum()
    var_x  = (w_s * dx ** 2).sum()
    var_z  = (w_s * dz ** 2).sum()
    W = cov_xz ** 2 / (var_x * var_z)

    pvalue = _royston_pvalue(W, n_eff)
    return W, pvalue


def _royston_pvalue(W, n):
    """
    Royston (1992) p-value approximation for the Shapiro-Wilk W statistic.

    Parameters
    ----------
    W : float   Observed W statistic.
    n : int     Effective sample size (3 <= n <= 5000).

    Returns
    -------
    float
        Approximate p-value (probability of observing W this small under H0).
    """
    W = float(W)
    n = int(n)

    # n = 3: exact closed-form result
    if n == 3:
        p = (6.0 / np.pi) * (np.arcsin(np.sqrt(W)) - np.arcsin(np.sqrt(0.75)))
        return max(float(p), 1e-99)

    y = np.log(1.0 - W)
    u = np.log(n)

    if 4 <= n <= 11:
        # Royston (1992): small-n uses a gamma shift before the normal transform.
        gamma = 0.459 * n - 2.273
        mu = (-1.5861
              + (-0.31082) * u
              + (-0.083751) * u ** 2
              +  0.0038915 * u ** 3)
        lu = np.log(u)
        log_sigma = (-0.4803
                     + (-0.082676) * lu
                     +  0.0030302 * lu ** 2)
        sigma = np.exp(log_sigma)
        z = (y - gamma - mu) / sigma
    else:
        # n >= 12: Royston (1992) polynomial approximations
        # mu: polynomial in log(n)
        mu = (-1.5861
              + (-0.31082) * u
              + (-0.083751) * u ** 2
              +  0.0038915 * u ** 3)
        # sigma: exp(polynomial in log(log(n)))
        lu = np.log(u)
        log_sigma = (-0.4803
                     + (-0.082676) * lu
                     +  0.0030302 * lu ** 2)
        sigma = np.exp(log_sigma)
        z = (y - mu) / sigma

    p = 1.0 - stats.norm.cdf(z)
    return max(float(p), 1e-99)


########################################################################
########################################################################
def empirical_metadata(data: np.ndarray, weights: np.ndarray, num_bins: int = 256, bias=False) -> dict[str, any]:
    """Compute a set of empirical statistics for metadata."""
    data = np.array(data).flatten()
    weights = np.array(weights).flatten()
    weights = weights/np.sum(weights)
    n = len(data)
    
    if len(data) != len(weights):
        raise ValueError(f'Length of data and weights must be equal, not {len(data)}, {len(weights)}')
    
    metadata = {}
    metadata['n'] = n
    
    Ws = [weights, np.ones_like(weights)/len(weights)]
    labels = ['', '_uw']
    for W, label in zip(Ws, labels):
        # basic metrics
        mean = np.sum(data*W)/np.sum(W)
        var = np.sum(W * (data-mean)**2)
        std = np.sqrt(var)
        skew = weighted_skew(data, W, bias=bias)
        kurt = weighted_kurtosis(data, W, bias=bias)
        
        metadata[f'mean{label}'] = mean
        # metadata[f'variance{label}'] = var
        # metadata[f'stdev{label}'] = std
        metadata[f'coeffvar{label}'] = std/mean
        metadata[f'skewness{label}'] = skew
        metadata[f'kurtosis{label}'] = kurt
        
        # entropy estimate via histogram/density (weighted). Shannon entropy
        hist, bin_edges = np.histogram(data, bins=num_bins, weights=W, density=True)
        hist = np.maximum(hist, 1e-16)
        probs = hist / hist.sum()
        entr = float(stats.entropy(probs))
        metadata[f'entropy{label}'] = entr

        # conservative mode count using smoothed histogram local maxima
        mode_count_est = estimate_maxima(data, weights=W, gran=1_000)
        metadata[f'mode_count_est{label}'] = mode_count_est

        # find the proportion of data classified as an outlier (farther than 1.5*IQR from the IQR)
        xcdf, ycdf, func = weighted_ecdf(data, W)
        q1, q3 = np.interp([0.25, 0.75], ycdf, xcdf)
        iqr = q3-q1
        outliers_lo = np.array([(x, w) for (x, w) in zip(data, W) if x < q1-1.5*iqr])
        outliers_hi = np.array([(x, w) for (x, w) in zip(data, W) if x > q3+1.5*iqr])
        if len(outliers_lo) == 0:
            wt_outliers_lo = 0
        else:
            wt_outliers_lo = np.sum(outliers_lo[:,1])
        if len(outliers_hi) == 0:
            wt_outliers_hi = 0
        else:
            wt_outliers_hi = np.sum(outliers_hi[:,1])
    
        metadata[f'weight_outliers{label}'] = wt_outliers_lo + wt_outliers_hi
        
        # Calculate strength of normal and lognormal fits
        metadata[f'fit_norm_SW{label}'] = shapiro_wilk_weighted(data, W)[0]
        metadata[f'fit_lognorm_SW{label}'] = shapiro_wilk_weighted(np.log(data), W)[0]
        # norm_results = weighted_distance_norm(data, W)
        # log_results = weighted_distance_norm(np.log(data), W)
        # for res, nln in zip([norm_results, log_results], ['norm', 'lognorm']):
        #     for method in res.keys():
        #         metadata[f'fit_{nln}_{method}{label}'] = res[method]

    # Calculate the Wasserstein metric between weighted and unweighted data
    metadata['w_v_uw_wasserstein'] = wasserstein1_weighted(data, data, weights, np.ones_like(data), unitless=False)
    
    # sort dictionary
    metadata = {k: metadata[k] for k in sorted(metadata)}
    return metadata


def NestedDictValues(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from NestedDictValues(v)
        else:
            yield v
########################################################################
def weighted_ecdf(data, weights=None, kind='previous'):
    """
    INPUT
    data     array of data points
    weights  array of weights (same length as data)
    
    OUTPUT
    xcdf     data values ordered in increasing value
    ycdf     corresponding y-values to the xcdf for the CDF function
    func     a function you can use to project onto whatever x values you need
    """
    data = np.array(data).flatten()
    if weights is None: weights = np.ones_like(data)
    weights = np.array(weights).flatten()
    if len(data) != len(weights): raise ValueError(f'data and weights must be the same length instead of {len(data)} and {len(weights)}')
    
    idx = np.argsort(data)
    xcdf, weights = data[idx], weights[idx]
    ycdf = np.cumsum(weights) / np.sum(weights)
    
    xcdf = np.concatenate([[-np.inf], xcdf, [np.inf]])
    ycdf = np.concatenate([[0], ycdf, [1]])
    func = scipy.interpolate.interp1d(xcdf, ycdf, kind=kind)

    return xcdf, ycdf, func
########################################################################
def estimate_maxima(data, weights=None, gran=1_000):
    """
    INPUT:
        data    array of data
        gran    number of data points in plot used to find local maxima
    
    OUTPUT:
        nmodes      the height of ymaxima minus the height of yminima where max(ymaxima)=1
    """
    if weights is None: weights=np.ones_like(data)
    bw = weighted_bw(data, weights, bw_method='scott')
    kde = gaussian_kde(data, bw_method=1.0, weights=weights)
    bw_base = (kde.covariance**0.5)[0][0]
    kde.set_bandwidth(bw/bw_base)
    xvals = np.linspace(min(data)-3*bw, max(data)+3*bw, gran)
    yvals = kde.evaluate(xvals)
    inds_max = argrelextrema(yvals, np.greater)
    inds_min = argrelextrema(yvals, np.less)
    ymaxima = yvals[list(inds_max)].flatten()
    yminima = yvals[list(inds_min)].flatten()
    nmodes = (np.sum(ymaxima)-np.sum(yminima))/np.max(ymaxima)

    return nmodes
########################################################################
def weighted_kurtosis(data, weights, bias=False):
    """
    INPUTS:
        data      array of data
        weights   array of weights (same length as data)
    
    OUTPUTS:
        kurt      skew calculated according to Fisher-Pearson coefficient of skewness
    """
    data = np.array(data).flatten()
    weights = np.array(weights).flatten()
    
    if len(data) != len(weights): raise ValueError(f'data and weights must be the same length, instead of {len(data)} and {len(weights)}')
    
    mean = np.sum(data*weights)
    m2 = np.sum(weights*(data-mean)**2)
    m4 = np.sum(weights*(data-mean)**4)
    n = len(data)

    if bias is False:
        g2 = n**2*((n+1)*m4-3*(n-1)*m2**2)/((n-1)*(n-2)*(n-3)) * (n-1)**2/(n**2*m2**2)
    else:
        g2 = m4/m2**2-3

    return g2
########################################################################
def weighted_skew(data, weights, bias=False):
    """
    Fisher-Pearson coefficient of skewness
    INPUTS:
        data      array of data
        weights   array of weights (same length as data)
    
    OUTPUTS:
        skew      skew calculated according to Fisher-Pearson coefficient of skewness
    """
    data = np.array(data).flatten()
    weights = np.array(weights).flatten()
    
    if len(data) != len(weights): raise ValueError(f'data and weights must be the same length, instead of {len(data)} and {len(weights)}')
    
    mean = np.sum(data*weights)
    m2 = np.sum(weights*(data-mean)**2)
    m3 = np.sum(weights*(data-mean)**3)
    n = len(data)
    g1 = (m3/m2**1.5)
    if bias is False:
        g1 = g1*np.sqrt(n*(n-1))/(n-2) #second part of this is the bias
    
    return g1
########################################################################
def wasserstein1_weighted(samples_p, samples_q, weights_p=None, weights_q=None, unitless=False):
    """
    Compute the Wasserstein-1 distance between two weighted empirical distributions.
    
    Parameters
    ----------
    samples_p : array-like
    samples_q : array-like
    weights_p : array-like, defaults to uniform
    weights_q : array-like, defaults to uniform
    
    Returns
    -------
    float : W1 distance
    """

    # format samples
    samples_p = np.asarray(samples_p, dtype=float)
    samples_q = np.asarray(samples_q, dtype=float)

    # normalize weights
    if weights_p is None:
        weights_p = np.ones_like(samples_p)
    weights_p = np.asarray(weights_p, dtype=float)
    
    if weights_q is None:
        weights_q = np.ones_like(samples_q)
    weights_q = np.asarray(weights_q, dtype=float)

    weights_p = weights_p / np.sum(weights_p)
    weights_q = weights_q / np.sum(weights_q)

    # result
    wass1 = wasserstein_distance(samples_p, samples_q, weights_p, weights_q)

    
    if unitless is True:
        std = weighted_std(np.concatenate([samples_p, samples_q]), np.concatenate([weights_p, weights_q]))
        wass1 = wass1 / std
    elif unitless is False:
        pass
    elif type(unitless) is float:
        wass1 = wass1 / unitless
    else:
        raise ValueError(f'unitless must be True, False, or a float, instead of {unitless}')

    return wass1

########################################################################
def wasserstein2_weighted(samples_p, samples_q, weights_p=None, weights_q=None, unitless=False):
    """
    Compute the Wasserstein-2 distance between two weighted empirical distributions.
    
    Parameters
    ----------
    samples_p : array-like
    samples_q : array-like
    weights_p : array-like, defaults to uniform
    weights_q : array-like, defaults to uniform
    
    Returns
    -------
    float : W2 distance
    """

    # format samples
    samples_p = np.asarray(samples_p, dtype=float)
    samples_q = np.asarray(samples_q, dtype=float)

    # normalize weights
    if weights_p is None:
        weights_p = np.ones_like(samples_p)
    weights_p = np.asarray(weights_p, dtype=float)
    
    if weights_q is None:
        weights_q = np.ones_like(samples_q)
    weights_q = np.asarray(weights_q, dtype=float)

    weights_p = weights_p / np.sum(weights_p)
    weights_q = weights_q / np.sum(weights_q)

    # Sort both distributions by sample value
    idx_p = np.argsort(samples_p)
    idx_q = np.argsort(samples_q)
    samples_p, weights_p = samples_p[idx_p], weights_p[idx_p]
    samples_q, weights_q = samples_q[idx_q], weights_q[idx_q]

    # Build cumulative weight arrays (right edge of each step)
    cdf_p = np.cumsum(weights_p)
    cdf_q = np.cumsum(weights_q)

    # Merge all CDF knots into a single grid
    u_grid = np.concatenate([[0], cdf_p, cdf_q, [1]])
    u_grid = np.unique(u_grid)

    # Evaluate quantile functions at midpoints of each interval
    u_mid = (u_grid[:-1] + u_grid[1:]) / 2
    qp = samples_p[np.searchsorted(cdf_p, u_mid, side='left').clip(0, len(samples_p)-1)]
    qq = samples_q[np.searchsorted(cdf_q, u_mid, side='left').clip(0, len(samples_q)-1)]

    # Integrate analytically: each interval has constant integrand
    interval_widths = np.diff(u_grid)
    wass2 = np.sum((qp - qq)**2 * interval_widths)
    wass2 = np.sqrt(wass2)

    
    if unitless is True:
        std = weighted_std(np.concatenate([samples_p, samples_q]), np.concatenate([weights_p, weights_q]))
        wass2 = wass2 / std**2
    elif unitless is False:
        pass
    elif type(unitless) is float:
        wass2 = wass2 / unitless
    else:
        raise ValueError(f'unitless must be True, False, or a float, instead of {unitless}')

    return wass2


# # --- Example usage ---
# rng = np.random.default_rng(42)

# samples_p = rng.normal(loc=0, scale=1, size=200)
# samples_q = rng.normal(loc=1, scale=1.5, size=300)

# # Random weights, normalized to sum to 1
# weights_p = rng.dirichlet(np.ones(200))
# weights_q = rng.dirichlet(np.ones(300))

# w2 = wasserstein2_weighted(samples_p, samples_q, weights_p, weights_q)
# print(f"W2 distance: {w2:.4f}")
########################################################################
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def weighted_var(x, w):
    xbar = weighted_mean(x, w)
    return np.sum(w * (x - xbar)**2) / np.sum(w)

def weighted_distance_norm(x, w, gran=1001):
    """
    Weighted statistical tests against a normal distribution

    Parameters
    ----------
    x    : array_like
           Data samples.
    w    : array_like
           Non-negative weights. Need not be normalized.
    gran : integer
           Granularity of the plots used to calculate difference between CDFs

    Returns
    -------
    A2 : float
        Weighted Anderson-Darling statistic.

    Notes
    -----
    The Anderson-Darling test statistic is derived from the 
    weighted squared difference between the empirical cumulative
    distribution function (Fn(x)) and the hypothesized CDF (F(x)), 
    with higher weight on the distribution tails. This core concept
    is used to calculate the weighted AD-statistic, so it does
    not perfectly match up with the typical, simplified AD
    calculation.

    """

    x = np.asarray(x)
    w = np.asarray(w)
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative")

    # Normalize weights
    w = w / np.sum(w)

    # Sort by x
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    
    # empirical model
    xecdf, yecdf, fecdf = weighted_ecdf(x, w)
    
    # theoretical model
    mean = weighted_mean(x, w)
    std = np.sqrt(weighted_var(x, w))
    
    # set bounds
    lo = np.min([np.min(x), mean-3*std])
    hi = np.max([np.max(x), mean+3*std])
    xplot = np.linspace(lo, hi, gran)
    
    # calculate input functions
    Fn = fecdf(xplot) # empirically observed observations
    F = norm.cdf(xplot, mean, std) # model distribution
    ymodl = norm.pdf(xplot, mean, std)
    F = F / np.max(F)

    
    results = {}
    # Anderson-Darling
    ind_lo = np.absolute(xplot-(mean-3*std)).argmin()
    ind_hi = np.absolute(xplot-(mean+3*std)).argmin()
    Fn_AD = Fn[ind_lo:ind_hi]
    F_AD = F[ind_lo:ind_hi]
    test, y2int = 'AD', (Fn_AD-F_AD)**2 / (F_AD*(1-F_AD))
    xad = xplot[ind_lo:ind_hi]
    totals = cumulative_trapezoid(y2int, xad, initial=0)
    totals = totals[np.all([np.isfinite(totals), ~np.isnan(totals)], axis=0)]
    results[test] = totals[-1]
    
    # Cramer-von Mises
    test, y2int = 'CVM', (Fn-F)**2,
    totals = cumulative_trapezoid(y2int, xplot, initial=0)
    totals = totals[np.all([np.isfinite(totals), ~np.isnan(totals)], axis=0)]
    results[test] = totals[-1]
    
    # Wasserstein-1
    test, y2int = 'W1', np.abs(F-Fn)
    results[test] = wasserstein1_weighted(x, xplot, w, ymodl, unitless=False)
    
    # Wasserstein-2 approximation using CDFs
    test, y2int = 'W2', np.abs(F-Fn)**2
    results[test] = wasserstein2_weighted(x, xplot, w, ymodl, unitless=False)
    
    # Energy Distance
    test = 'ED'
    results[test] = energy_distance(xplot, xplot, Fn, F)
    
    # Kolmogorov-Smirnov
    test = 'KS'
    results[test] = np.max(np.abs(F-Fn))
    
    results = {i: results[i] for i in sorted(results.keys())}
    return results




################################################################################################################
################################################################################################################


def weighted_quantile(X, W, x, output='perc2val'):
    """
    Given a set of values with weights, returns a value from a percentage, or a percentage from a value
    INPUTS:
    X       values
    W       weights
    x       desired percentage or value
    output  'perc2val' or 'val2perc' (as string)
    
    OUTPUT:
    value or percentage, depending on 'output' variable
    """
    cdf = W.copy()    
    y_cdf, cdf = zip(*sorted(zip(np.append(X, [0]), np.append(cdf, [0])), key=lambda x: x[0]))
    y_cdf, cdf = np.array(y_cdf), np.array(cdf)
    for i in range(len(cdf)-1):
        cdf[i+1] = cdf[i] + W[i]
    cdf[0] = 0
    cdf[-1] = 1
    y_cdf[0] = min(X)
    
    if output == 'perc2val':
        perc2val = scipy.interpolate.interp1d(cdf, y_cdf, assume_sorted=False)
        return perc2val(x)
    
    elif output == 'val2perc':
        val2perc = scipy.interpolate.interp1d(y_cdf, cdf, assume_sorted=False)
        return val2perc(x)
    
    else:
        raise ValueError("output must be 'perc2val' or 'val2perc'")

################################################################################################################
################################################################################################################

def weighted_bw(X, W, bw_method='silverman'):
    """
    Calculates bandwidth using Silverman's or Scott's method, adjusted for weighted data.

    Parameters
    ----------
    X : array_like
        Data values.
    W : array_like
        Weights for each data point.
    bw_method : str
        'silverman' or 'scott'

    Returns
    -------
    bw : float
        Bandwidth estimate (scalar)
    """
    X = np.array(X).flatten()
    W = np.array(W).flatten()
    W = W/sum(W)
        
    if len(X) != len(W):
        raise ValueError(f'Length of X and W must be equal: {len(X)}, {len(W)}')
    
    std = weighted_std(X, W)
    iqr = weighted_quantile(X, W, 0.75, output='perc2val') - weighted_quantile(X, W, 0.25, output='perc2val')
    if iqr == 0 or not np.isfinite(iqr):
        iqr = std * 1.34

    #effective sample size (n_eff) used since len(X) may overstate the sample size with uneven weights
    n_eff = (np.sum(W))**2 / np.sum(W**2)
    # n_eff = len(X)
    
    if bw_method=='silverman':
        bw = 0.9 * np.min([std, iqr/1.34]) * n_eff**-0.2
    elif bw_method=='scott':
        bw = 1.06 * std * n_eff**-0.2
    else:
        raise ValueError(f'bw_method must be silverman or scott instead of {bw_method}')
    
    return bw

################################################################################################################
################################################################################################################

def weighted_std(X, W, BW=None):
    """
    Computes the standard deviation of a weighted dataset, optionally adding kernel variance.

    Parameters
    ----------
    X : array_like
        Data values.
    W : array_like
        Weights (same length as X).
    BW : array_like or None
        Optional bandwidths per data point (standard deviations of kernels).
        If None, no kernel variance is added.

    Returns
    -------
    std : float
        Weighted standard deviation (including kernel contribution if BW is provided).
    """
    X = np.array(X).flatten()
    W = np.array(W).flatten()
    W = W/np.sum(W)
    if BW is None:
        BW = np.zeros_like(W)
    else:
        BW = np.array(BW).flatten()
    
    if not(len(X) == len(W) == len(BW)):
        raise ValueError("X, W, and BW must have the same length.")

    mean = np.sum(X * W)
    var_data = np.sum(W * (X - mean)**2)
    var_kernel = np.sum(W * BW**2)
    std = np.sqrt(var_data + var_kernel)
    return std

################################################################################################################
################################################################################################################

def bw_dirichlet(X, alpha, BW_factors, Wrun, bw_method='silverman'):
    """
    Estimates bandwidths for a weighted KDE with variable bandwidths per point.
    Uses a baseline estimate (weighted_bw), and adjusts only if necessary to meet target KDE std.

    Parameters
    ----------
    X : array_like
        Data points.
    alpha : array_like
        Concentration parameter vector (used for target weights).
    BW_factors : array_like
        Relative bandwidth multipliers (mean = 1).
    Wrun : array_like
        Sampled weights from Dirichlet distribution.
    bw_method : str
        Method for baseline bandwidth estimate ('silverman' or 'scott').

    Returns
    -------
    BW : ndarray
        Bandwidths to use for KDE (standard deviations of each kernel).
    """
    # Flatten and normalize
    X = np.array(X).flatten()
    W = np.array(alpha).flatten()
    Wrun = np.array(Wrun).flatten()
    BW_factors = np.array(BW_factors).flatten()

    W = W / np.sum(W)
    Wrun = Wrun / np.sum(Wrun)
    BW_factors = BW_factors / np.mean(BW_factors)

    # Step 1: Compute target variance using reference weights (W)
    bw_base = weighted_bw(X, W, bw_method)  # based on reference weights
    std_target = weighted_std(X, W, BW_factors * bw_base)
    var_target = std_target**2

    # Step 2: Try using weighted_bw with Wrun
    bw = weighted_bw(X, Wrun, bw_method)
    std_data = weighted_std(X, Wrun)
    var_data = std_data**2

    std_kernel = bw * BW_factors
    var_kernel = np.sum(Wrun * std_kernel**2)

    var_kde_guess = var_data + var_kernel
    
    #verify weighted_std and manual calculation are getting the same answer
    a = np.round(var_kde_guess, 4)
    b = np.round(weighted_std(X, Wrun, bw * BW_factors)**2, 4)
    if  a != b :
        raise ValueError(f'Should be 1: {a} / {b}')

    # Step 3: Accept guess if it meets or exceeds target
    if var_kde_guess >= var_target:
        # print(f'bw_base / bw: {bw_base} / {bw}')
        # print(f'bw_dirichlet var_data: {var_data}')
        # print(f'bw_dirichlet var_kernel: {var_kernel}')
        # print(f'bw_dirichlet var_target: {var_target}')
        # print('')
        return bw * BW_factors  # Accept the guess
    
    # Step 4: Otherwise, solve for exact bw to hit target    
    else:
        bw = np.sqrt((var_target - var_data) / np.sum(Wrun * BW_factors**2))
        std_kernel = bw * BW_factors
        var_kernel = np.sum(Wrun * std_kernel**2)
        # print(f'bw_base / bw: {bw_base} / {bw}')
        # print(f'bw_dirichlet (else) var_data: {var_data}')
        # print(f'bw_dirichlet (else) var_kernel: {var_kernel}')
        # print(f'bw_dirichlet (else) var_target: {var_target}')
        # print('')
        if np.round(var_kernel + var_data, 4) < np.round(var_target, 4):
            raise ValueError(f'bw_dirichlet function didnt work. var_kernel+var_data={var_kernel+var_data}; var_target={var_target}')
        return bw * BW_factors

    raise ValueError(f'Failed to return a value')