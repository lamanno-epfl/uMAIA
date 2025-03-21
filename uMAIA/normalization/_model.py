from jax import pure_callback, random
import jax.numpy as jnp
import optax
import tqdm
import numpy as np
import numpyro
from numpyro import handlers
from numpyro.contrib.funsor import config_enumerate, infer_discrete
import numpyro.distributions as dist
from numpyro.infer import SVI, TraceEnum_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta

@config_enumerate
def model(data, mask, covariate_vector=None, D_matrix_ones=None, 
          priors_hyperparameters=None, delta_v_dist='gaussian',flex_mean=0.5):
    N, S, V = data.shape
    K = 2

    if covariate_vector == None:
        covariate_vector = np.zeros((S,), dtype=np.int8)
    n_cov = len(np.unique(covariate_vector))
    covariate_vector = jnp.array(covariate_vector)

    # D_matrix construction
    if D_matrix_ones == None:
        D_matrix_ones = jnp.zeros((n_cov,S))
        for i, c in enumerate(covariate_vector):
            D_matrix_ones = D_matrix_ones.at[c, i].set(1)
    else:
        n_cov = D_matrix_ones.shape[0]
    D_matrix_ones_unsqueeze = jnp.expand_dims(D_matrix_ones, axis=(2,3))

    # The Model
    V_plate = numpyro.plate('v_plate', V,dim=-1)
    S_plate = numpyro.plate('s_plate', S, dim=-2)
    D_plate = numpyro.plate('d_plate', len(data), dim=-3)
    C_plate = numpyro.plate('c_plate', n_cov, dim=-4)
    
    with S_plate:
        b_gamma = numpyro.sample('b_gamma', dist.Uniform(jnp.full((S, 1), -2),
                                                       jnp.full((S, 1), 2))
        )

        sigma_s = numpyro.sample('sigma_s', dist.Exponential(jnp.full((S, 1), 
                                                                      priors_hyperparameters['rate0_sigma_s'])))
        
      
    with V_plate:
        if delta_v_dist=='gamma':
            delta_v = numpyro.sample('delta_v', 
                                    dist.Normal(jnp.full((V), 3.0) ,
                                                jnp.full((V), 0.8))
                                  )
        else:
            delta_v = numpyro.sample('delta_v', 
                                    dist.Gamma(jnp.full((V), 5.0) ,
                                                jnp.full((V), 2.0))
                                  ) #dist.Gamma(3, 0.8)
        with C_plate:
            # delta = numpyro.sample('delta', 
            #                     dist.Normal(jnp.full((n_cov,1, 1, V), 3.0) ,
            #                                 jnp.full((n_cov,1, 1, V), 0.8))
            #                   ) 
            delta = numpyro.sample('delta', 
                                dist.Normal(jnp.full((n_cov,1, 1, V), delta_v) ,
                                            jnp.full((n_cov,1, 1, V), 0.25))
                              ) 

         
        locs = numpyro.sample('locs', dist.Normal(jnp.full((V,), priors_hyperparameters['mu0_b']) ,
                                               jnp.full((V,), 1))
                          )
        
        scale1 = numpyro.sample('scale1', dist.Uniform(jnp.full((V,), priors_hyperparameters['lowerBound_scale1']),
                                                    jnp.full((V,), 0.5))
                            )
        
        sigma_v = numpyro.sample('sigma_v',
                              dist.Exponential(jnp.full((V,), 
                                                        priors_hyperparameters['rate0_sigma_v'])))
        
        b_lambda = numpyro.sample('b_lambda',
                                dist.Uniform(jnp.full((V,), -2.0),
                                             jnp.full((V,),  2.0)))
        
    delta_ = jnp.sum(jnp.sum(delta * D_matrix_ones_unsqueeze, axis=-4), axis=-2)


    
    with S_plate:
        with V_plate:
            weights = numpyro.sample('weights',
                                  dist.Dirichlet(0.5 * jnp.ones(K)))
            batch_error = numpyro.sample('error', dist.Uniform(jnp.full((S,V), -flex_mean),
                                                    jnp.full((S,V), flex_mean)))
    
            batch_term = b_lambda*b_gamma
            batch_term = batch_term + batch_error
            
    with (S_plate, V_plate, D_plate):
        with handlers.mask(mask = mask):
            assignment = numpyro.sample('assignment', dist.Categorical(weights),
                                   # infer={"enumerate":"parallel"}
                                )

            obs = numpyro.sample('obs', dist.Normal(jnp.where(assignment==0, 
                                                        locs, 
                                                        locs + delta_ + batch_term), 
                                                    jnp.where(assignment==0,
                                                        scale1, 
                                                        sigma_s + sigma_v)),
                    obs=data) 

@config_enumerate
def model_rank2(data, mask, covariate_vector=None, D_matrix_ones=None, priors_hyperparameters=None, flex_mean=0.5):
    N, S, V = data.shape
    K = 2

    if covariate_vector == None:
        covariate_vector = np.zeros((S,), dtype=np.int8)
    n_cov = len(np.unique(covariate_vector))
    covariate_vector = jnp.array(covariate_vector)

    # D_matrix construction
    if D_matrix_ones == None:
        D_matrix_ones = jnp.zeros((n_cov,S))
        for i, c in enumerate(covariate_vector):
            D_matrix_ones = D_matrix_ones.at[c, i].set(1)
    else:
        n_cov = D_matrix_ones.shape[0]
    D_matrix_ones_unsqueeze = jnp.expand_dims(D_matrix_ones, axis=(2,3))

    # The Model
    V_plate = numpyro.plate('v_plate', V,dim=-1)
    S_plate = numpyro.plate('s_plate', S, dim=-2)
    D_plate = numpyro.plate('d_plate', len(data), dim=-3)
    C_plate = numpyro.plate('c_plate', n_cov, dim=-4)
    
    with S_plate:
        b_gamma = numpyro.sample('b_gamma', dist.Uniform(jnp.full((S, 1), -2),
                                                       jnp.full((S, 1), 2))
        )
        b_gamma2 = numpyro.sample('b_gamma2', dist.Uniform(jnp.full((S, 1), -2),
                                                       jnp.full((S, 1), 2))
        )

        sigma_s = numpyro.sample('sigma_s', dist.Exponential(jnp.full((S, 1), 
                                                                      priors_hyperparameters['rate0_sigma_s'])))
        
      
    with V_plate:
        delta_v = numpyro.sample('delta_v', 
                                dist.Normal(jnp.full((V), 3.0) ,
                                            jnp.full((V), 0.5)) #0.8
                              ) 
        with C_plate:
            # delta = numpyro.sample('delta', 
            #                     dist.Normal(jnp.full((n_cov,1, 1, V), 3.0) ,
            #                                 jnp.full((n_cov,1, 1, V), 0.8))
            #                   ) 
            delta = numpyro.sample('delta', 
                                dist.Normal(jnp.full((n_cov,1, 1, V), delta_v) ,
                                            jnp.full((n_cov,1, 1, V), 0.1)) # 0.25
                              ) 

         
        locs = numpyro.sample('locs', dist.Normal(jnp.full((V,), priors_hyperparameters['mu0_b']) ,
                                               jnp.full((V,), 1))
                          )
        
        scale1 = numpyro.sample('scale1', dist.Uniform(jnp.full((V,), priors_hyperparameters['lowerBound_scale1']),
                                                    jnp.full((V,), 0.5))
                            )
        
        sigma_v = numpyro.sample('sigma_v',
                              dist.Exponential(jnp.full((V,), 
                                                        priors_hyperparameters['rate0_sigma_v'])))
        
        b_lambda = numpyro.sample('b_lambda',
                                dist.Uniform(jnp.full((V,), -2.0),
                                             jnp.full((V,),  2.0)))
        b_lambda2 = numpyro.sample('b_lambda2',
                                dist.Uniform(jnp.full((V,), -2.0),
                                             jnp.full((V,),  2.0)))
        

    delta_ = jnp.sum(jnp.sum(delta * D_matrix_ones_unsqueeze, axis=-4), axis=-2)


    
    with S_plate:
        with V_plate:
            weights = numpyro.sample('weights',
                                  dist.Dirichlet(0.5 * jnp.ones(K)))
            batch_error = numpyro.sample('error', dist.Uniform(jnp.full((S,V), -flex_mean),
                                                    jnp.full((S,V), flex_mean)))
    
            batch_term1 = b_lambda*b_gamma
            batch_term2 = b_lambda2*b_gamma2

            batch_term = batch_term1 + (1.0 * batch_term2) + batch_error
            
    with (S_plate, V_plate, D_plate):
        with handlers.mask(mask = mask):
            assignment = numpyro.sample('assignment', dist.Categorical(weights),
                                   # infer={"enumerate":"parallel"}
                                )

            obs = numpyro.sample('obs', dist.Normal(jnp.where(assignment==0, 
                                                        locs, 
                                                        locs + delta_ + batch_term), 
                                                    jnp.where(assignment==0,
                                                        scale1, 
                                                        sigma_s + sigma_v)),
                    obs=data) 
