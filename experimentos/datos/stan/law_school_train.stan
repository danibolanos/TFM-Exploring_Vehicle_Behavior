data {
  int<lower = 0> N; // number of observations
  int<lower = 0> C; // number of covariates
  matrix[N, C]   A; // sensitive variables
  real           GPA[N]; // GPA
  int            LSAT[N]; // LSAT
  real           FYA[N]; // FYA
}


transformed data { 
 vector[C] zero_C = rep_vector(0,C);
 vector[C] one_C = rep_vector(1,C);
}


parameters {
  vector[N] K;

  real b_G;
  real wK_G;
  real b_L;
  real wK_L;
  real wK_F;
  
  vector[C] wA_G;
  vector[C] wA_L;
  vector[C] wA_F;
  
  
  real<lower=0> sigma2_G;
}


transformed parameters  {
 // Population standard deviation (a positive real number derived from variance)
 real<lower=0> sigma_G = sqrt(sigma2_G);
}


model {
  // don't have data about this
  K ~ normal(0, 1);
  b_G  ~ normal(0, 1);
  wK_G ~ normal(0, 1);
  b_L  ~ normal(0, 1);
  wK_L ~ normal(0, 1);
  wK_F ~ normal(0, 1);

  wA_G ~ normal(zero_C, one_C);
  wA_L ~ normal(zero_C, one_C);
  wA_F ~ normal(zero_C, one_C);

  sigma2_G ~ inv_gamma(1, 1);

  // have data about these
  GPA ~ normal(b_G + K * wK_G + A * wA_G, sigma_G);
  LSAT ~ poisson(exp(b_L + K * wK_L + A * wA_L));
  FYA ~ normal(K * wK_F + A * wA_F, 1);
}
