import scipy.stats as sps
from scipy.interpolate import interp1d
import numpy as np
import jax.numpy as jnp

def transform(x, mask, svi_results):
    
    weights = svi_results.params['weights_auto_loc']
    locs = svi_results.params['locs_auto_loc']
    scale1 = svi_results.params['scale1_auto_loc']
    sigma_v = svi_results.params['sigma_v_auto_loc']
    b_lambda = svi_results.params['b_lambda_auto_loc']
    b_gamma = svi_results.params['b_gamma_auto_loc']
    delta = svi_results.params['delta_auto_loc']
    sigma_s = svi_results.params['sigma_s_auto_loc']
    error = svi_results.params['error_auto_loc']
    delta_ = svi_results.params['delta_']
    loc0_delta = svi_results.params['loc0_delta']
    
    x_tran = np.zeros_like(x)
    M̂1 = locs 
    Ŝs1 = scale1
    Ŝs2 = sigma_v

    for v in range(len(locs)):
        #icdf = make_inv_cdf(M̂1[v], M̂2[v], Ŝs1 + _sigmav1[v], Ŝs2 + _sigmav2[v])
        
        for s in range(len(sigma_s)):
            try:
                M̂2 = locs + delta_[s,v]
            except:
                M̂2 = locs + delta[v]
            icdf = make_inv_cdf(M̂1[v], M̂2[v], Ŝs1[v], Ŝs2[v])
            #x_tran[:,s, v][mask[:,s,v]] = transform_byicdf(x[:,s,v][mask[:,s,v]], _mu1[s, v], _mu2[s, v], _sigmas1[s] + _sigmav1[v], _sigmas2[s] + _sigmav2[v],icdf)
            
            try:
                x_tran[:,s, v][mask[:,s,v]] = transform_byicdf(x[:,s,v][mask[:,s,v]], locs[v], locs[v] + delta_[s,v] + b_gamma[s]*b_lambda[v] + error[s,v] + loc0_delta
                                                        , scale1[v], sigma_v[v] + sigma_s[s] ,icdf)
            except:
                x_tran[:,s, v][mask[:,s,v]] = transform_byicdf(x[:,s,v][mask[:,s,v]], locs[v], locs[v] + delta[v,] + b_gamma[s]*b_lambda[v] + error[s,v] + loc0_delta
                                                        , scale1[v], sigma_v[v] + sigma_s[s] ,icdf)
        
    
    return x_tran



def make_inv_cdf(mu1, mu2, sigma1, sigma2,
                 resolution=2000, rel_range=6):
    mu_min, mu_max = min(mu1, mu2), max(mu1, mu2)
    
    sigma_min, sigma_max = min(sigma1, sigma2), max(sigma1, sigma2)
    domain = np.linspace(mu_min - rel_range*sigma_max,
                         mu_max + rel_range*sigma_max,
                         resolution)
    cdf_vals = (sps.norm.cdf(domain, mu1, sigma1) + sps.norm.cdf(domain, mu2, sigma2)) / 2.
    # function to convert the cummulative density into the domain (the real values)
    icdf = interp1d(cdf_vals, domain, bounds_error=False, fill_value="extrapolate", copy=True)
    return icdf


def cdf_mixture(x, mu1, mu2, sigma1, sigma2):
    return (sps.norm.cdf(x, mu1, sigma1) + sps.norm.cdf(x, mu2, sigma2)) / 2.


def transform_byicdf(x, mu1x, mu2x, sigma1x, sigma2x, icdfy):
    """Usage: pass as icdfy the function returned by make_inv_cdf
    
    
    icdfy: a function that maps a cdf from a reference to a domain.
    For example:
    xnew = np.linspace(0, 10, 1000)
    icdf = make_inv_cdf(-4, 0, 2, 2)
    newx = transform_byicdf(xnew, 4, 8, 1, 1, icdf)
    """
    return icdfy(cdf_mixture(x,mu1x, mu2x, sigma1x, sigma2x))