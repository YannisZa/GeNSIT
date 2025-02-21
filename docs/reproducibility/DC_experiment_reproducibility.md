# R Squared Analysis

```
clear; gensit run ./data/inputs/configs/generic/experiment_r_squared.toml \
-et RSquared_Analysis -d DC -gt 200029.0 -nt 3 -nw 12
```

```
clear; gensit summarise -dn DC/r_squared \
-d RSquared_Analysis_LowNoise_16_04_2024_18_01_20 \
-e maxr2 "r2[arg_max]" \
-e alpha "alpha_range[arg_max[0]]" \
-e beta "beta_range[arg_max[1]]" \
-ea r2 -el np \
-ea "alpha_min=outputs.config['experiments'][0]['grid_ranges']['alpha']['min']" \
-ea "alpha_max=outputs.config['experiments'][0]['grid_ranges']['alpha']['max']" \
-ea "alpha_n=outputs.config['experiments'][0]['grid_ranges']['alpha']['n']" \
-ea "alpha_range=np.linspace(alpha_min,alpha_max,alpha_n,endpoint=True)" \
-ea "beta_min=outputs.config['experiments'][0]['grid_ranges']['beta']['min']" \
-ea "beta_max=outputs.config['experiments'][0]['grid_ranges']['beta']['max']" \
-ea "beta_n=outputs.config['experiments'][0]['grid_ranges']['beta']['n']" \
-ea "beta_range=np.linspace(beta_min,beta_max,beta_n,endpoint=True)" \
-ea "arg_max=np.unravel_index(r2.squeeze().argmax(), np.shape(r2.squeeze()))" \
-k cost_matrix -k destination_attraction_ts -k bmax -fe R2
```

# NN

```
clear; gensit run ./data/inputs/configs/DC/experiment1_nn_disjoint.toml -et SIM_NN -nt 6 -nw 4
```

```
clear; gensit run ./data/inputs/configs/DC/experiment1_nn_disjoint.toml -et NonJointTableSIM_NN -nt 6 -nw 4
```

```
clear; gensit run ./data/inputs/configs/DC/experiment1_nn_joint.toml -et JointTableSIM_NN -nt 6 -nw 4
```

# MCMC

```
clear; gensit run ./data/inputs/configs/DC/experiment1_mcmc_low_noise.toml -et SIM_MCMC -nt 12 -nw 1 -n 20000
```

```
clear; gensit run ./data/inputs/configs/DC/experiment1_mcmc_high_noise.toml -et SIM_MCMC -nt 12 -nw 1 -n 50000
```

```
clear; gensit run ./data/inputs/configs/DC/experiment1_mcmc_low_noise.toml -et JointTableSIM_MCMC -nt 12 -nw 1 -n 20000
```

```
clear; gensit run ./data/inputs/configs/DC/experiment1_mcmc_high_noise.toml -et JointTableSIM_MCMC -nt 12 -nw 1 -n 20000
```

# Comparison methods

```
clear; gensit run ./data/inputs/configs/DC/experiment1_mcmc_high_noise.toml -et JointTableSIM_MCMC -nt 12 -nw 1 -n 20000

clear; gensit run ./data/inputs/configs/DC/experiment1_mcmc_low_noise.toml -et JointTableSIM_MCMC -nt 12 -nw 1 -n 20000

clear; gensit run ./data/inputs/configs/DC/vanilla_comparisons.toml -et RandomForest_Comparison -nt 15 -nw 2 -rf 'mini_region_features.npy' -ttl '_doubly_and_cell_constrained_mini_region_features'

clear; gensit run ./data/inputs/configs/DC/vanilla_comparisons.toml -et RandomForest_Comparison -nt 15 -nw 2 -rf 'region_features.npy' -ttl '_doubly_and_cell_constrained_all_region_features'

clear; gensit run ./data/inputs/configs/DC/vanilla_comparisons.toml -et GBRT_Comparison -nt 15 -nw 2 -rf 'mini_region_features.npy' -ttl '_doubly_and_cell_constrained_mini_region_features'

clear; gensit run ./data/inputs/configs/DC/vanilla_comparisons.toml -et GBRT_Comparison -nt 15 -nw 2 -rf 'region_features.npy' -ttl '_doubly_and_cell_constrained_all_region_features'

clear; gensit run ./data/inputs/configs/DC/vanilla_comparisons.toml -et XGBoost_Comparison -nt 15 -nw 2 -rf 'mini_region_features.npy' -ttl '_doubly_and_cell_constrained_mini_region_features'





# aquifer

# csic40
clear; gensit run ./data/inputs/configs/DC/vanilla_comparisons.toml -et GraphAttentionNetwork_Comparison -nt 15 -nw 2 -rf 'mini_region_features.npy' -ttl '_doubly_and_cell_constrained_mini_region_features' -n 10000

# csic41
clear; gensit run ./data/inputs/configs/DC/vanilla_comparisons.toml -et XGBoost_Comparison -nt 15 -nw 1 -rf 'region_features.npy' -ttl '_doubly_and_cell_constrained_all_region_features'

# csic43

# csic46
clear; gensit run ./data/inputs/configs/DC/vanilla_comparisons.toml -et GraphAttentionNetwork_Comparison -nt 15 -nw 2 -rf 'region_features.npy' -ttl '_doubly_and_cell_constrained_all_region_features' -n 10000
```

# Summaries and Metrics


## SRMSEs


### Intensity

#### GeNSIT

Get SRMSEs for intensity samples and my experiments:

-et SIM_NN -et NonJointTableSIM_NN -et JointTableSIM_NN \
```
clear; gensit summarise -dn DC/exp1 -et SIM_NN \
-el np -el MathUtils -el xr \
-e intensity_srmse_test_mean "intensity_srmse_test_by_seed.mean('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_test_std "intensity_srmse_test_by_seed.std('seed',dtype='float64',skipna=True)" \
-ea intensity \
-ea "intensity_mean=intensity.groupby('seed').mean('iter',dtype='float64')" \
-ea "intensity_srmse_all_by_seed=intensity_mean.groupby('seed').map(MathUtils.srmse,ground_truth=outputs.inputs.data.ground_truth_table)" \
-ea "intensity_srmse_train_by_seed=intensity_mean.groupby('seed').map(MathUtils.srmse,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.train_cells_mask)" \
-ea "intensity_srmse_test_by_seed=intensity_mean.groupby('seed').map(MathUtils.srmse,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.test_cells_mask)" \
-btt 'iter' 10000 9 10000 \
-k sigma -k type -k name -k title -fe total_constrained_intensity_SRMSEs -nw 1
```

#### Vanilla comparisons

Get SRMSEs for intensity samples and vanilla comparison experiments:

-et XGBoost_Comparison -et GraphAttentionNetwork_Comparison -et GBRT_Comparison -et RandomForest_Comparison \
```
clear; gensit summarise -dn DC/comparisons \
-et GBRT_Comparison -et RandomForest_Comparison \
-el np -el MathUtils -el xr \
-e intensity_srmse_all_mean "intensity_srmse_all.mean('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_all_std "intensity_srmse_all.std('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_train_mean "intensity_srmse_train.mean('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_train_std "intensity_srmse_train.std('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_test_mean "intensity_srmse_test.mean('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_test_std "intensity_srmse_test.std('seed',dtype='float64',skipna=True)" \
-ea intensity \
-ea "intensity_mean=intensity.groupby('seed').mean('iter',dtype='float64')" \
-ea "intensity_srmse_all=intensity_mean.groupby('seed').map(MathUtils.srmse,ground_truth=outputs.inputs.data.ground_truth_table)" \
-ea "intensity_srmse_train=intensity_mean.groupby('seed').map(MathUtils.srmse,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.train_cells_mask)" \
-ea "intensity_srmse_test=intensity_mean.groupby('seed').map(MathUtils.srmse,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.test_cells_mask)" \
-k type -k title -fe intensity_SRMSEs -nw 2
```

### Table 

#### GeNSIT
Get SRMSEs for table samples for my experiments:

-et NonJointTableSIM_NN -et JointTableSIM_NN \
-cs "da.loss_name==str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])" \
```
clear; gensit summarise -dn DC/exp1 -et JointTableSIM_NN \
-el np -el MathUtils -el xr \
-e table_srmse_all_mean "table_srmse_all_by_seed.mean('seed',dtype='float64',skipna=True)" \
-e table_srmse_all_std "table_srmse_all_by_seed.mean('seed',dtype='float64',skipna=True)" \
-e table_srmse_train_mean "table_srmse_train_by_seed.mean('seed',dtype='float64',skipna=True)" \
-e table_srmse_train_std "table_srmse_train_by_seed.std('seed',dtype='float64',skipna=True)" \
-e table_srmse_test_mean "table_srmse_test_by_seed.mean('seed',dtype='float64',skipna=True)" \
-e table_srmse_test_std "table_srmse_test_by_seed.std('seed',dtype='float64',skipna=True)" \
-ea table \
-ea "table_mean=table.groupby('seed').mean('iter',dtype='float64')" \
-ea "table_srmse_all_by_seed=table_mean.groupby('seed').map(MathUtils.srmse,ground_truth=outputs.inputs.data.ground_truth_table)" \
-ea "table_srmse_train_by_seed=table_mean.groupby('seed').map(MathUtils.srmse,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.train_cells_mask)" \
-ea "table_srmse_test_by_seed=table_mean.groupby('seed').map(MathUtils.srmse,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.test_cells_mask)" \
-btt 'iter' 10000 9 10000 \
-cs "da.loss_name==str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])" \
--slice -k sigma -k type -k name -k title -fe table_SRMSEs -nw 1
```


## Sorensen Similarity Index

### Intensity

#### GeNSIT
Get SSI for intensity samples and my experiments:

```
clear; gensit summarise \
-dn DC/exp1 \
-et SIM_NN -et NonJointTableSIM_NN -et JointTableSIM_NN \
-el np -el MathUtils -el xr \
-e intensity_ssi_train_mean "intensity_ssi_train.mean('seed',dtype='float64',skipna=True)" \
-e intensity_ssi_train_std "intensity_ssi_train.std('seed',dtype='float64',skipna=True)" \
-e intensity_ssi_test_mean "intensity_ssi_test.mean('seed',dtype='float64',skipna=True)" \
-e intensity_ssi_test_std "intensity_ssi_test.std('seed',dtype='float64',skipna=True)" \
-ea intensity \
-ea "intensity_mean=intensity.groupby('seed').mean('iter',dtype='float64')" \
-ea "intensity_ssi_train=intensity_mean.groupby('seed').map(MathUtils.ssi,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.train_cells_mask)" \
-ea "intensity_ssi_test=intensity_mean.groupby('seed').map(MathUtils.ssi,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.test_cells_mask)" \
-btt 'iter' 10000 10 100000 \
-k sigma -k type -k name -k title -fe total_constrained_intensity_SSIs -nw 1
```

#### Vanilla Comparisons
Get SSI for intensity samples and vanilla comparison experiments:

-et XGBoost_Comparison -et GraphAttentionNetwork_Comparison -et GBRT_Comparison -et RandomForest_Comparison \
```
clear; gensit summarise -dn DC/comparisons \
-et GBRT_Comparison -et RandomForest_Comparison \
-el np -el MathUtils -el xr \
-e intensity_ssi_train_mean "intensity_ssi_train.mean('seed',dtype='float64',skipna=True)" \
-e intensity_ssi_train_std "intensity_ssi_train.std('seed',dtype='float64',skipna=True)" \
-e intensity_ssi_test_mean "intensity_ssi_test.mean('seed',dtype='float64',skipna=True)" \
-e intensity_ssi_test_std "intensity_ssi_test.std('seed',dtype='float64',skipna=True)" \
-ea intensity \
-ea "intensity_mean=intensity.groupby('seed').mean('iter',dtype='float64')" \
-ea "intensity_ssi_train=intensity_mean.groupby('seed').map(MathUtils.ssi,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.train_cells_mask)" \
-ea "intensity_ssi_test=intensity_mean.groupby('seed').map(MathUtils.ssi,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.test_cells_mask)" \
-k type -k title -fe intensity_SSIs -nw 2
```

### Table

#### GeNSIT
Get SSI for table samples for my experiments:

clear; gensit summarise -dn DC/exp1 -et NonJointTableSIM_NN -et JointTableSIM_NN \
-cs "da.loss_name==str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])" \
```
clear; gensit summarise -dn DC/exp1 -et JointTableSIM_NN \
-el np -el MathUtils -el xr \
-e table_ssi_train_mean "table_ssi_train.mean('seed',dtype='float64',skipna=True)" \
-e table_ssi_train_std "table_ssi_train.std('seed',dtype='float64',skipna=True)" \
-e table_ssi_test_mean "table_ssi_test.mean('seed',dtype='float64',skipna=True)" \
-e table_ssi_test_std "table_ssi_test.std('seed',dtype='float64',skipna=True)" \
-ea table \
-ea "table_mean=table.mean('iter',dtype='float64')" \
-ea "table_ssi_train=table_mean.groupby('seed').map(MathUtils.ssi,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.train_cells_mask)" \
-ea "table_ssi_test=table_mean.groupby('seed').map(MathUtils.ssi,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.test_cells_mask)" \
-btt 'iter' 10000 9 100000 \
-cs "da.loss_name==str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])" \
--slice -k sigma -k type -k name -k title -fe table_SSIs -nw 1
```


## Coverage Probabilities

### Intensity

#### GeNSIT

Get coverage probabilities for intensity samples and my experiments:

-et SIM_NN -et JointTableSIM_NN -et NonJointTableSIM_NN
```
clear; gensit summarise -dn DC/exp1 -et SIM_NN -et JointTableSIM_NN -et NonJointTableSIM_NN \
-el np -el MathUtils -el xr \
-e intensity_cp_train_mean "xr.apply_ufunc(roundint, 100*intensity_cp_train.mean(['origin','destination'],skipna=True)).mean('seed',dtype='float64',skipna=True)" \
-e intensity_cp_train_std "xr.apply_ufunc(roundint, 100*intensity_cp_train.mean(['origin','destination'],skipna=True)).std('seed',dtype='float64',skipna=True)" \
-e intensity_cp_test_mean "xr.apply_ufunc(roundint, 100*intensity_cp_test.mean(['origin','destination'],skipna=True)).mean('seed',dtype='float64',skipna=True)" \
-e intensity_cp_test_std "xr.apply_ufunc(roundint, 100*intensity_cp_test.mean(['origin','destination'],skipna=True)).std('seed',dtype='float64',skipna=True)" \
-ea intensity \
-ea "cp_func=MathUtils.coverage_probability" \
-ea "roundint=MathUtils.roundint" \
-ea "region_masses=0.99" \
-ea "intensity_cp_train=intensity.stack(id=['iter']).groupby('seed').map(cp_func,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.train_cells_mask,region_mass=region_masses)" \
-ea "intensity_cp_test=intensity.stack(id=['iter']).groupby('seed').map(cp_func,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.test_cells_mask,region_mass=region_masses)" \
-btt 'iter' 10000 9 10000 \
-k type -k title -fe total_constrained_intensity_CoverageProbabilities -nw 1
```

#### Vanilla Comparisons

Get coverage probabilities for intensity samples and vanilla algorithm comparisons:

-et GBRT_Comparison -et GraphAttentionNetwork_Comparison -et RandomForest_Comparison -et XGBoost_Comparison \
```
clear; gensit summarise  -dn DC/comparisons \
-et GBRT_Comparison -et RandomForest_Comparison \
-el np -el MathUtils -el xr \
-e intensity_cp_train_mean "xr.apply_ufunc(roundint, 100*intensity_cp_train.mean(['origin','destination'],skipna=True)).mean('seed',dtype='float64',skipna=True)" \
-e intensity_cp_train_std "xr.apply_ufunc(roundint, 100*intensity_cp_train.mean(['origin','destination'],skipna=True)).std('seed',dtype='float64',skipna=True)" \
-e intensity_cp_test_mean "xr.apply_ufunc(roundint, 100*intensity_cp_test.mean(['origin','destination'],skipna=True)).mean('seed',dtype='float64',skipna=True)" \
-e intensity_cp_test_std "xr.apply_ufunc(roundint, 100*intensity_cp_test.mean(['origin','destination'],skipna=True)).std('seed',dtype='float64',skipna=True)" \
-ea intensity \
-ea "cp_func=MathUtils.coverage_probability" \
-ea "roundint=MathUtils.roundint" \
-ea "region_masses=0.99" \
-ea "intensity_cp_train=intensity.stack(id=['iter']).groupby('seed').map(cp_func,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.train_cells_mask,region_mass=region_masses)" \
-ea "intensity_cp_test=intensity.stack(id=['iter']).groupby('seed').map(cp_func,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.test_cells_mask,region_mass=region_masses)" \
-btt 'iter' 10000 9 10000 \
-btt 'iter' 0 1 10000 \
-k type -k title -fe intensity_CoverageProbabilities -nw 2
```

### Table

#### GeNSIT

Get coverage probabilities for table samples and my experiments:

-et JointTableSIM_NN -et NonJointTableSIM_NN
```
clear; gensit summarise -dn DC/exp1 -et NonJointTableSIM_NN \
-el np -el MathUtils -el xr \
-e table_cp_train_mean "xr.apply_ufunc(roundint, 100*table_cp_train.mean(['origin','destination'],skipna=True)).mean('seed',dtype='float64',skipna=True)" \
-e table_cp_train_std "xr.apply_ufunc(roundint, 100*table_cp_train.mean(['origin','destination'],skipna=True)).std('seed',dtype='float64',skipna=True)" \
-e table_cp_test_mean "xr.apply_ufunc(roundint, 100*table_cp_test.mean(['origin','destination'],skipna=True)).mean('seed',dtype='float64',skipna=True)" \
-e table_cp_test_std "xr.apply_ufunc(roundint, 100*table_cp_test.mean(['origin','destination'],skipna=True)).std('seed',dtype='float64',skipna=True)" \
-ea table \
-ea "cp_func=MathUtils.coverage_probability" \
-ea "roundint=MathUtils.roundint" \
-ea "region_masses=0.99" \
-ea "table_cp_train=table.stack(id=['iter']).groupby('seed').map(cp_func,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.train_cells_mask,region_mass=region_masses)" \
-ea "table_cp_test=table.stack(id=['iter']).groupby('seed').map(cp_func,ground_truth=outputs.inputs.data.ground_truth_table,mask=outputs.inputs.data.test_cells_mask,region_mass=region_masses)" \
-btt 'iter' 10000 9 10000 \
-k type -k title -fe table_CoverageProbabilities -nw 1
```


# Plots

## Figure 5

```
clear; gensit plot spatial empty -x residual_mean_colsums_spatial  \
-dn DC/comparisons -et GraphAttentionNetwork_Comparison_UnsetNoise__doubly_and_cell_constrained_all_region_features \
-el np -el MathUtils -o ./data/outputs/ \
-e residual_mean_total_spatial "residual_mean_colsums.sum('destination')" \
-e residual_mean_colsums_spatial "residual_mean_colsums.to_dataframe(name='data',dim_order=['destination'])" \
-ea intensity \
-ea "intensity_mean=intensity.groupby('seed').mean('iter',dtype='float64')" \
-ea "residual_mean_colsums=intensity_mean.groupby('seed').map(lambda x: (x-outputs.inputs.data.ground_truth_table).where(outputs.inputs.data.test_cells_mask,drop=True)).mean('seed',skipna=True).sum('origin')" \
-xlab 'Longitude' -ylab 'Latitute' -at 'GMEL' \
-fs 10 10 -ff ps -ft 'figure5/mean_residual' -cm RdYlBu_r -vmid 0.0 -la 0 0 \
-ats 16 -ylr 90 -yts 12 0 -xts 12 0 -yls 16 0 -xls 16 0 -nw 1
```



```
clear; gensit plot spatial geoshow -x residual_mean_colsums_spatial  \
-pdd ./data/outputs/DC/comparisons/paper_figures/figure5/ \
-el np -el MathUtils \
-e residual_mean_colsums_spatial "residual_mean_colsums.to_dataframe(name='data',dim_order=['destination'])" \
-ea intensity \
-ea "intensity_mean=intensity.groupby('seed').mean('iter',dtype='float64')" \
-ea "residual_mean_colsums=intensity_mean.groupby('seed').map(lambda x: (x-outputs.inputs.data.ground_truth_table).where(outputs.inputs.data.test_cells_mask,drop=True)).mean('seed',skipna=True).sum('origin')" \
-xlab 'Longitude' -ylab 'Latitute' -at 'GMEL' \
-fs 10 10 -ff ps -ft 'mean_residual' -cm RdYlBu_r -vmid 0.0 -la 0 0 \
-ats 18 -ylr 90 -yts 12 0 -xts 12 0 -nw 1
```