data {
  int<lower=0> N; 
  int<lower=0> K;
  int<lower=1,upper=K> county[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[K] alpha;
  vector[K] beta;
  real<lower=0,upper=100> sigma;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  real mu_a;
  real mu_b;
} 
model {
  y ~ normal(alpha[county] + beta[county].*x, sigma);
  alpha ~ normal(mu_a,sigma_a);
  beta ~ normal(mu_b,sigma_b);
  mu_a ~ normal(0,10);
  mu_b ~ normal(0,10);
  sigma_a ~ cauchy(0,1);
  sigma_b ~ cauchy(0,1);
}

