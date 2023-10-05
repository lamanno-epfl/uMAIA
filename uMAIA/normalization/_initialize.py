import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import invgamma, norm, lognorm, gamma
from scipy.stats import kstest
import tqdm
import jax.numpy as jnp


def initialize(x, mask, subsample=True, lower_bound_scale1=0.05, idx=None):
    
    if subsample==True:
        if idx==None:
            average_num_pixels = int(np.mean(mask.sum(axis=0)[:,0]))
            idx = np.random.choice(np.arange(2000, average_num_pixels), size=2000, replace=False)
            idx = np.append(np.arange(500), idx)

        x = x[idx,:,:]
        mask = mask[idx,:,:]
        
    mu, sigma = _init_gmm(x, mask)
    
    priors_hyperparameters, init_values = _estimate_priors(mu, sigma, lower_bound_scale1, shape=x.shape)
    
    init_state = {"prior_hyperparams": priors_hyperparameters,
            "init_values": init_values}
    
    return init_state
    
    

def _init_gmm(x, mask, metric='BIC'):
    _, _, V = x.shape
    
    num_components = [1, 2]
    mu_ = np.zeros((V, 2))
    sigma_ = np.zeros((V, 2))
    weights_ = np.zeros((V, 2))


    for v in tqdm.tqdm(range(V), desc='GMM Initialization'):

        data = x[:, :, v][mask[:, :, v]].reshape(-1,1)
        lowest_score = np.infty
        best_gmm = None
        best_n = -1
        
        if len(data) > 2:
            for n in num_components:
                gmm = GMM(n_components = n).fit(data)
                
                if metric=="BIC":
                    score = gmm.bic(data)

                elif metric=="AIC":
                    score = gmm.aic(data)
                    
                else:
                    return "Incorrect Metric for Initialization. Metric could be AIC or BIC."
                    
                    
                if score < lowest_score:
                    lowest_score = score
                    best_gmm = gmm
                    best_n = n
                
            if best_n == 2:
                mu_[v,:] = best_gmm.means_.reshape(-1)
                sigma_[v, :] = best_gmm.covariances_.reshape(-1)
                weights_[v, :] = best_gmm.weights_.reshape(-1)
            else:
                mu_[v, :] = np.repeat(best_gmm.means_, 2)
                sigma_[v, :] = np.repeat(best_gmm.covariances_, 2)
                weights_[v, :] = np.repeat(best_gmm.weights_, 2)
            
            
    sorted_indices = np.argsort(mu_)
    for v in range(V):
        mu_[v, :] =  mu_[v, sorted_indices[v, :]]
        sigma_[v, :] = sigma_[v, sorted_indices[v, :]]
        weights_[v, :] = weights_[v, sorted_indices[v, :]]
        
        
    mu_[:, 0] = np.percentile(mu_[:, 0], 25)
    return (mu_, sigma_)




def _estimate_priors(mu_, sigma_, lower_bound_scale1=0.05, shape=None):
    N, S, V = shape
    K = 2
    
    params_sigmaF = gamma.fit(sigma_[:, 1])
    params_delta = gamma.fit(mu_[:, 1] - mu_[:, 0])
    
    mu0_b = np.mean(mu_[:, 0])

    concentration0_sigma = params_sigmaF[0]
    rate_sigma0_f = 1/params_sigmaF[2]
    rate_sigma0_s = 0.5 * rate_sigma0_f
    rate_sigma0_v = 0.5 * rate_sigma0_f

    concentration0_delta = params_delta[0]
    rate0_delta = 1/params_delta[2]
    loc0_delta = params_delta[1]
    
    priors_hyperparameters = {
        'mu0_b': mu0_b,
        'concentration0_sigma': concentration0_sigma,
        'rate0_sigma_v': rate_sigma0_v,
        'rate0_sigma_s': rate_sigma0_s,
        'concentration0_delta': concentration0_delta,
        'rate0_delta': rate0_delta,
        'loc0_delta': loc0_delta,
        'lowerBound_scale1': lower_bound_scale1,
    }
    
    ub_scale1 = np.median(np.sqrt(sigma_[:, 1]))
    lb_scale1 = ub_scale1 / 2

    init_locs = jnp.ones(V) * mu0_b
    init_weights = jnp.ones((S, V, K)) / K
    init_scale1 = jnp.ones(V) * lower_bound_scale1 #np.median(np.sqrt(sigma_[:, 0]))
    init_sigmav = jnp.ones(V) * lb_scale1
    init_delta = jnp.expand_dims(jnp.array(mu_[:, 1] - mu_[:, 0]), axis=(0,1,2))

    init_values = {
        "weights": init_weights,
        "locs": init_locs,
        "scale1": init_scale1,
        "sigma_v": init_sigmav,
        "delta": init_delta,
        "b_lambda": jnp.ones(V) * 0.1,
        "b_gamma": jnp.ones((S,1)) * 0.1,
        "sigma_s": jnp.ones((S,1)) * lb_scale1,
        "error": jnp.ones((S,V)) * 0.01
        }


    return (priors_hyperparameters, init_values)

