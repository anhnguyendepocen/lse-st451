data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n,p] X;
  vector[n] y;
}
parameters {
  vector[p] beta;
  real<lower=0> sigma;
}
model {
  beta ~ normal(0,sigma);
  sigma ~ cauchy(0,1);
  y ~ normal(X*beta, sigma);
}
