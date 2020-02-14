// The input data is a vector 'y' of length 'N' consisting of binary variables.
// and the a vector x for the covariate 'nodes_detected'
data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
  vector[N] x;
}

// The parameters accepted by the model. Our model
// contains the beta coefficients as parameters.
parameters {
  vector[2] beta;
}

// The model to be estimated. 
model {
  beta ~ normal(0,100);
  //target += bernoulli_logit_lpmf(y | beta[1] + beta[2] * x);
  for (n in 1:N)
    //y[n] ~ bernoulli(inv_logit(beta[1] + beta[2] * x[n]));
    y[n] ~ bernoulli_logit(beta[1]+ beta[2] * x[n]);
}

