data {
  int<lower=0> n; 
  int<lower=0> N;
  vector[n] y;
} 
parameters {
  real<lower=0> iv;
  real<lower=0> sigma;
  real mu;
} 

transformed parameters{
  real<lower=0> v;
  v=1/iv;
}
model {
  iv ~ gamma(0.5*N,0.5*N*sigma*sigma);
  y ~ normal(mu, v);
}

