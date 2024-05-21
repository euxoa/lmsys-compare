import polars as pl
import numpy as np
import jax.numpy as jnp
import jax
# from jax import config
# config.update("jax_enable_x64", True)
import jaxopt
import argparse


def parse_args():                                                                                                                           
     parser = argparse.ArgumentParser(description="Compare LLMs on the basis of LMSYS leaderboard data.")                                                           
     parser.add_argument("lmsys_input_data", type=str, help="LMSYS input data in their JSON format")                                               
     parser.add_argument("--nmodels", "-n", type=int, default=9999999, help="Number of models with most data to include (default: all)")
     parser.add_argument("--language_regex", "-l", type=str, default="")   
     parser.add_argument('--code', dest='code', action='store_true', help='Enable code')                                                         
     parser.add_argument('--no-code', dest='code', action='store_false', help='Disable code')                                                    
     parser.set_defaults(code=None)                                                                                                                                               
     parser.add_argument("--models", "-m", type=str, default="", help="Models to report (regex, default: '')") 
     parser.add_argument("--min_mse", "-e", type=float, default=0.2, help="Only report models with MSE of skill below this value.")                                                                                                                                                
     args = parser.parse_args()
     args.allowed_code_values = (True, False) if args.code is None else (args.code,)
     return args                                                                                                                             

args = parse_args()

# Prefiltering and aggregation of the model comparison data. 
# Note that there are actually many covariates on token lengths, prompt quality, duplication etc now.
# Some of them should be used for filtering maybe, or even better, multidimensional modelling.
d_orig = (pl.read_json(args.lmsys_input_data).
          with_columns(model_a = pl.col("model_a").str.replace("^###.*A: ", ""), 
                       model_b = pl.col("model_b").str.replace("^###.*B: ", "")).
          unnest('num_tokens_info').
          filter(pl.col("winner").is_in(("model_a", "model_b"))).
          filter(pl.col("is_code").is_in(args.allowed_code_values)).
          filter(pl.col("user_tokens")>50).
          filter(pl.col("language").str.contains(args.language_regex)).
          group_by(["model_a", "model_b", "winner"]).
          agg(count = pl.len()).
          sort(pl.col("count"), descending = True)
)

n_models = args.nmodels

# Pick only n_model models and set canonical ids here. 
# (To select models, find total amount of comparisons per model.)
models = (pl.concat((d_orig.select(model="model_a", count="count"), 
                     d_orig.select(model="model_b", count="count"))).
          group_by("model").
          agg(pl.col("count").sum()).
          sort(pl.col("count"), descending=True).
          head(n_models).
          with_columns(id = pl.col("model").rank("dense")-1))

# Join model ids to data, and a bit of cleanup. 
d_reduced = (d_orig.
             join(models.select(model_a="model", id_a="id"), on="model_a").
             join(models.select(model_b="model", id_b="id"), on="model_b").
             filter(pl.col("id_a") != pl.col("id_b")))

# Final data for estimation. Mainly folds the upper-right triangle
# of the comparison matrix to the lower left and sums up.
d_m = (d_reduced.
        with_columns((pl.col("winner") == "model_a").alias("a_wins"),
                     (pl.col("id_a") > pl.col("id_b")).alias("swapp")).
        with_columns(pl.when(pl.col("swapp")).then("id_b").otherwise("id_a").alias("id1"),
                     pl.when(pl.col("swapp")).then("id_a").otherwise("id_b").alias("id2"),
                     pl.when(pl.col("swapp")).then(pl.col("a_wins").not_()).otherwise(pl.col("a_wins")).alias("win1")).
        group_by(["id1", "id2", "win1"]).agg(pl.col("count").sum()).
        pivot(index=["id1", "id2"], columns="win1", values="count").
        rename({'true' : 'win1', 'false' : 'win2'}).
        with_columns(pl.col("win1").fill_null(strategy="zero")).
        with_columns(pl.col("win2").fill_null(strategy="zero")).
        with_columns((pl.col('win1') + pl.col('win2')).alias("tot")))

# Convert from Pandas to a dictionary for JAX.
d_mj = {i : jnp.array(d_m[i]) for i in d_m.columns}

if False:
    # Write the cleaned-up winning counts to a file for potential estimation
    # with other methods (such as R and Stan.)
    d_m.write_csv("d_m.csv", separator=';')

def cost(params, d_mj):
    """
    Calculates the deviance (likelihood) of a potentially multidimensional IRT-like
    binomial logit model. 
    """
    id1, id2, win1, win2 = d_mj['id1'], d_mj['id2'], d_mj['win1'], d_mj['win2']
    logit_diff = jnp.sum(params[id1] - params[id2], -1)
    lp0 = jax.nn.log_sigmoid( logit_diff)
    lp1 = jax.nn.log_sigmoid(-logit_diff)
    # Note that if you regularize, you lose the pinv() trick later on calculating standard deviations.
    llh = jnp.sum(win1 * lp0 + win2 * lp1) #- .0 * jnp.sum(jnp.sum(params**2, 0) * jnp.arange(params.shape[-1]))
    # Return deviance.
    return -2*llh

n_dim = 1 # We want a one-dimensional IRT here. ;)
opt = jaxopt.LBFGS(fun=cost, maxiter=20) # FIXME: doesn't check convergence, although n_it<10 is typically enough.
params = 0.01*jax.random.normal(jax.random.key(42), (min(n_models, models.shape[0]), n_dim), dtype='float32')
res = opt.run(params, d_mj = d_mj)
params, state = res
# By quick tests, the sd below seems to be right, even a bit conservative in practice. 
# Note that if you set n_dim>1, the formula below only gives the errors for the 
# first skill parameters, conditional on others!
# The posterior without regularization is degenerate as skills can all be transformed by \(x) x+const, 
# pinv() takes care of that. 
sd = jnp.sqrt(jnp.diag(jnp.linalg.pinv(.5*jax.hessian(cost)(params, d_mj=d_mj)[:, 0, :, 0], rcond=1e-5)))

o_frame = \
    models.sort("id").with_columns(
    pl.Series(name="skill", values=iter(params[:,0])),
    pl.Series(name="MSE_skill", values=iter(sd))
    ).sort("skill", descending=True).\
    filter((pl.col("MSE_skill") < args.min_mse)  &  
           (pl.col("count") > 10) & # i'm afraid of the rcond in the pinv() above
           pl.col("model").str.contains(args.models))

pl.Config.set_tbl_rows(1002)
print("Input:", args.lmsys_input_data)
print(o_frame)


# A look at eigenvalues of a multidimensional solution. 
# FOr some reason, these are super-collinear, i.e., one (underlying) dimension dominates.
if False:
    print("Eigenvalues, or square roots:", jax.numpy.linalg.svd(params)[1])

# Outer ufuncs, the JAX style. I guess newaxis would do as well?
# print(jax.vmap(jax.vmap(jnp.subtract, (None, 0)), (0, None))(a, b))

# A test. Note that skills are unique only up to a const. 
if False:
    trueparms = jnp.array([-0.5, 0, 0, 1, 2.])
    n_models, n_obs = len(trueparms), 1000
    wins = np.random.binomial(n_obs, jax.nn.sigmoid(trueparms[:, jnp.newaxis] - trueparms[jnp.newaxis, :]))
    losses = n_obs - wins
    d_mj = { k: jnp.array(v) 
            for k, v in zip(('id1', 'id2', 'win1', 'win2'), 
                            zip(*[(x, y, wins[x, y], losses[x, y]) 
                                for x, y in zip(*(jnp.ravel(i) for i in jnp.indices(wins.shape)))]))}

    n_dim = 1
    params = 0.01*jax.random.normal(jax.random.key(42), (n_models, n_dim), dtype='float32')
    opt = jaxopt.LBFGS(fun=cost, maxiter=100)
    res = opt.run(params, d_mj = d_mj)
    params, state = res
    print(params[:,0]) # note that these may be shifted (average to zero from regularization)
    # The sd here is for the one-dimensional case only. ;)
    print(jnp.sqrt(jnp.diag(jnp.linalg.pinv(.5*jax.hessian(cost)(params, d_mj=d_mj)[:, 0, :, 0], rcond=1e-3))))
    print(jnp.linalg.eigvals(.5*jax.hessian(cost)(params, d_mj=d_mj)[:, 0, :, 0]))


