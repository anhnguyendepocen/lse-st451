data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n,p] X;
  vector[n] y;
}
parameters {
  real<lower=0> s1;
  real<lower=0> s2;
  real<lower=0> t1;
  real<lower=0> t2;
  vector<lower=0>[p] ell1;
  vector<lower=0>[p] ell2;
  vector[p] z;
  //real<lower=0> tau;
  //vector<lower=0>[p] lambda;
  //real<lower=0> sigma;
}
transformed parameters{
  vector[p] beta;
  vector<lower=0>[p] lambda;
  real<lower=0> sigma;
  real<lower=0> tau;
  sigma = s1 .* sqrt(s2);
  lambda = ell1 .* sqrt(ell2);
  tau = t1 .* sqrt(t2);
  beta = z .* lambda*tau;
}
model {
  z ~ normal(0,1);
  s1 ~ normal(0, 1);
  s2 ~ inv_gamma (0.5, 0.5);
  t1 ~ normal(0, 1);
  t2 ~ inv_gamma (0.5, 0.5);
  ell1 ~ normal(0, 1);
  ell2 ~ inv_gamma (0.5, 0.5);
  //sigma ~ cauchy(0,1);
  //lambda ~ cauchy(0, 1);
  //tau ~ cauchy(0, 1);
  y ~ normal(X*beta, sigma);
}
