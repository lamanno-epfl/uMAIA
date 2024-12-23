from jax import random
import jax.numpy as jnp
import optax
import numpy as np
from numpyro import handlers
from numpyro.infer import SVI, TraceEnum_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta

from ._model import model, model_rank2
from ._initialize import initialize


def normalize(data, 
              mask,
              subsample=True,
              idx=None,
              init_state=None,
              optimizer=None, 
              loss=None, 
              num_steps=5000,
              covariate_vector=None,
              D_matrix_ones=None,
              flex_mean=0.5,
              delta_v_dist='gaussian',
              seed=42,
              rank=1):

    N, S, V = data.shape
    # initialise via GMM
    if init_state==None:
        init_state = initialize(data, mask, subsample=subsample, idx=idx)

    # retrieve covariate vector
    if covariate_vector == None:
        covariate_vector = np.zeros((S, ), dtype=np.int8)
    n_cov = len(np.unique(covariate_vector)) 
    covariate_vector = jnp.array(covariate_vector)

    if D_matrix_ones != None:
        n_cov = D_matrix_ones.shape[0]

    # modify delta to reflect covariates
    init_state['init_values']['delta'] = jnp.tile(init_state['init_values']['delta'], (n_cov, 1, 1, 1))
    
    if subsample==True:
        if idx==None:
            average_num_pixels = int(np.mean(mask.sum(axis=0)[:,0]))
            idx = np.random.choice(np.arange(2000, average_num_pixels), size=2000, replace=False)
            idx = np.append(np.arange(500), idx)

        data = data[idx,:,:]
        mask = mask[idx,:,:]
    
    
    
    
    
    if optimizer == None:
        optimizer = optax.adam(learning_rate=0.001, b1=0.95, b2=0.99)
    
    if loss == None:
        loss = TraceEnum_ELBO(max_plate_nesting=4)
    
    if rank==1:
        print('using rank 1')
        global_model = handlers.block(
            handlers.seed(model, random.PRNGKey(seed)),
            hide_fn=lambda site: site["name"]
            not in ['weights', 'locs', 'scale1', 'sigma_v', 'delta',
                    'b_lambda', 'b_gamma', 'sigma_s', 'error',
                    's_plate', 'v_plate', 'c_plate', 'd_plate'],
        )
    else:
        print('line 75: using rank 2')
        global_model = handlers.block(
            handlers.seed(model_rank2, random.PRNGKey(seed)),
            hide_fn=lambda site: site["name"]
            not in ['weights', 'locs', 'scale1', 'sigma_v', 'delta',
                    'b_lambda','b_lambda2', 'b_gamma','b_gamma2', 'sigma_s', 'error',
                    's_plate', 'v_plate', 'c_plate', 'd_plate'],
        )
        # add init values
        init_state['init_values']['b_lambda2'] =init_state['init_values']['b_lambda'].copy() * 1.1
        init_state['init_values']['b_gamma2'] =init_state['init_values']['b_gamma'].copy() * 1.1

    # set priors and initial states
    priors_hyperparameters = init_state["prior_hyperparams"]
    init_values = init_state["init_values"]

    # set the guide
    global_guide = AutoDelta(global_model,
                             init_loc_fn=init_to_value(values=init_values))
    
    if rank==1:
        global_svi = SVI(model, global_guide, optim=optimizer, loss=loss)
    elif rank==2:
        print('line 94: using rank 2')
        global_svi = SVI(model_rank2, global_guide, optim=optimizer, loss=loss)
    
    svi_result = global_svi.run(
            random.PRNGKey(0), num_steps, data, mask, covariate_vector=covariate_vector, D_matrix_ones=D_matrix_ones, priors_hyperparameters=priors_hyperparameters, flex_mean=flex_mean, delta_v_dist=delta_v_dist)
    
    
    # D_matrix
    if D_matrix_ones == None:
        D_matrix_ones = jnp.zeros((n_cov,S))
        for i, c in enumerate(covariate_vector):
            D_matrix_ones = D_matrix_ones.at[c, i].set(1)
    else:
        print('Using design matrix')
    D_matrix_ones_unsqueeze = jnp.expand_dims(D_matrix_ones, axis=(2,3))

    svi_result.params['delta_'] = jnp.sum(jnp.sum(svi_result.params['delta_auto_loc'] * D_matrix_ones_unsqueeze, axis=-4), axis=-2)
    svi_result.params['loc0_delta'] = priors_hyperparameters['loc0_delta']
    
    
    return svi_result