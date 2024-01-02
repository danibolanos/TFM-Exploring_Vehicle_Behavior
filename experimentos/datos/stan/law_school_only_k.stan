data {
  int<lower = 0> N; // number of observations
  int<lower = 0> C; // number of covariates
  matrix[N, C]   A; // sensitive variables
  real           GPA[N]; // GPA
  int            LSAT[N]; // LSAT
  
  real           b_G;
  real           wK_G;
  real           b_L;
  real           wK_L;
  
  vector[C]      wA_L;
  vector[C]      wA_G;
  
  real           sigma_G;
}


parameters {
  vector[N] K;
}


model {
  K ~ normal(0, 1);

  // have data about these
  GPA ~ normal(b_G + K * wK_G + A * wA_G, sigma_G); 
  LSAT ~ poisson(exp(b_L + K * wK_L + A * wA_L)); 
}
