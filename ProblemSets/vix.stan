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
  real<lower=0> kappa;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  mu ~ normal(0,100);
  sigma ~ normal(0,100);
  kappa ~ normal(0,100);
  for (i in 2:N)
    y[i] ~ student_t(3,y[i-1]+kappa*(mu-y[i-1]),sigma);
}


