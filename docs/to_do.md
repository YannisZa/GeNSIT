# Contingency table

## Implementation
  -  Read sample range in resize routine

## Tests
  - resize in test_2d_contingency_table_augmentation
  - 4x6 initialisation tests
  - Modify the following:
    degree_higher_proposal_addition_2x2_table
    degree_higher_proposal_2x2_table copy

# Markov basis

## Implementation
  - Use different data structure for storing markov bases (arrays are probably better than dictionaries)
    1. can lists of ndarrays be pickled for export?
    2. how fast can two objects be compared?

## Tests
  - update_and_augment tests

# Contingency Table MCMC

## Implementation
  - See if I can avoid copying f_prev altogether and just update f_prev on the fly

## Tests
  - table proposal degree higher step size support shrinkage for large tables (central and non-central hypergeometric)
  - degree_higher_proposal_2way_table_central_hypergeometric
  - retest every metropolis hastings acceptance ratio routine

# Spatial Interaction Model MCMC

## Tests
- Gradient of potential
- Hessian

## Speed up code

1. Write poisson likelihood and its gradient in C.
2. Replace f_to_df with another function

# Experiments

## Implementation
- Store samples every X iterations and keep only last sample in memory
