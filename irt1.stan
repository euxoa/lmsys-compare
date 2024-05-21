data {
  int<lower=0> N; // Number of matches
  int<lower=0> K; // Number of models
  int<lower=1, upper=K> model_a[N]; // Model indices for model_a
  int<lower=1, upper=K> model_b[N]; // Model indices for model_b
  int<lower=0> wins_a[N]; // Wins for model_a
  int<lower=0> wins_b[N]; // Wins for model_b
}

parameters {
  vector[K] skill; // Skill levels for each model
}

model {
  // Priors
  skill ~ normal(0, 2);
  mean(skill) ~ normal(0, .001);
  // Likelihood
  for (n in 1:N) {
    int total = wins_a[n] + wins_b[n];
    wins_a[n] ~ binomial_logit(total, skill[model_a[n]] - skill[model_b[n]]);
  }
}
