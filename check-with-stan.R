library(dplyr)
library(rstan)
# note tidyr and read there too

d_m <- readr::read_delim("d_m.csv")
  
stan_data <- 
  with(d_m, list(
    N = length(id1),
    K = max(c(id1, id2)) + 1,
    model_a = id1+1, model_b = id2+1, wins_a = win1, wins_b = win2))


fit <- stan("irt1.stan", data = stan_data, iter = 5000, chains = 1)

posterior_samples <- rstan::extract(fit)$skill
est <- data.frame(id = seq(0, stan_data$K - 1), 
                  skill_R = apply(posterior_samples, 2, mean), 
                  MSE_skill_R = apply(posterior_samples, 2, sd)) |> tidyr::as_tibble()

# Now that data frame could be given back to Python, or whatever. 
