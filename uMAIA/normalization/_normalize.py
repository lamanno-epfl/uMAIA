from jax import random
import jax.numpy as jnp
import optax
import numpy as np
from numpyro import handlers
from numpyro.infer import SVI, TraceEnum_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta

from ._model import model
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
              seed=42):
    
   
    if init_state==None:
        init_state = initialize(data, mask, subsample=subsample, idx=idx)
    
    if subsample==True:
        if idx==None:
            average_num_pixels = int(np.mean(mask.sum(axis=0)[:,0]))
            idx = np.random.choice(np.arange(2000, average_num_pixels), size=2000, replace=False)
            idx = np.append(np.arange(500), idx)

        data = data[idx,:,:]
        mask = mask[idx,:,:]
    
    
    priors_hyperparameters = init_state["prior_hyperparams"]
    init_values = init_state["init_values"]
    N, S, V = data.shape
    
    if optimizer == None:
        optimizer = optax.adam(learning_rate=0.001, b1=0.95, b2=0.99)
    
    if loss == None:
        loss = TraceEnum_ELBO(max_plate_nesting=4)
    
    
    global_model = handlers.block(
        handlers.seed(model, random.PRNGKey(seed)),
        hide_fn=lambda site: site["name"]
        not in ['weights', 'locs', 'scale1', 'sigma_v', 'delta',
                'b_lambda', 'b_gamma', 'sigma_s', 'error',
                's_plate', 'v_plate', 'c_plate', 'd_plate'],
    )
    
    global_guide = AutoDelta(global_model,
                             init_loc_fn=init_to_value(values=init_values))
    
    global_svi = SVI(model, global_guide, optim=optimizer, loss=loss)
    
    svi_result = global_svi.run(
            random.PRNGKey(0), num_steps, data, mask, covariate_vector=covariate_vector, priors_hyperparameters=priors_hyperparameters)
    
    
    
    if covariate_vector == None:
        covariate_vector = np.zeros((S, ), dtype=np.int8)
    n_cov = len(np.unique(covariate_vector)) 
    covariate_vector = jnp.array(covariate_vector)
    
    # D_matrix
    D_matrix_ones = jnp.zeros((n_cov,S))
    for i, c in enumerate(covariate_vector):
        D_matrix_ones = D_matrix_ones.at[c, i].set(1)
    D_matrix_ones_unsqueeze = jnp.expand_dims(D_matrix_ones, axis=(2,3))

    svi_result.params['delta_'] = jnp.sum(jnp.sum(svi_result.params['delta_auto_loc'] * D_matrix_ones_unsqueeze, axis=-4), axis=-2)
    svi_result.params['loc0_delta'] = priors_hyperparameters['loc0_delta']
    
    
    return svi_result