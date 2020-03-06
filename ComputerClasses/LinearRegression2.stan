data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n,p] X;
  vector[n] y;
}
parameters {
  vector[p] z;
  real<lower=0> s1;
  real<lower=0> s2;
  //vector[p] beta;
  //real<lower=0> sigma;
}
transformed parameters{
  vector[p] beta;
  real<lower=0> sigma;
  sigma = s1 .* sqrt(s2);
  beta = z*sigma;
}
model {
  z ~ normal(0,1);
  s1 ~ normal(0, 1);
  s2 ~ inv_gamma (0.5, 0.5);
  //beta ~ normal(0,1);
  //sigma ~ cauchy(0,1);
  y ~ normal(X*beta, sigma);
}
