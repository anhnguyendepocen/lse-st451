// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  real y[N];
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real mu;
  real<lower=0> sigma;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  mu ~ normal(0,1000);
  sigma ~ normal(0,100);
  //for (i in 1:N)
  // y[i] ~ normal(mu,sigma);
  target += normal_lpdf(y| mu, sigma);
}

