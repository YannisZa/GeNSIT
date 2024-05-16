# DC

## R Squared Analysis

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

## NN

```
clear; gensit run ./data/inputs/configs/DC/experiment1_nn_disjoint.toml -et SIM_NN -nt 6 -nw 1
```

```
clear; gensit run ./data/inputs/configs/DC/experiment1_nn_disjoint.toml -et NonJointTableSIM_NN -nt 8 -nw 3
```

```
clear; gensit run ./data/inputs/configs/DC/experiment1_nn_joint.toml -et JointTableSIM_NN -nt 8 -nw 3
```

## MCMC

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

## Comparison methods

```
clear; gensit run ./data/inputs/configs/DC/experiment1_mcmc_high_noise.toml -et JointTableSIM_MCMC -nt 12 -nw 1 -n 20000
clear; gensit run ./data/inputs/configs/DC/vanilla_comparisons.toml -et XGBoost_Comparison -nt 12 -nw 1 -n 20000
```

## Summaries and Metrics

### Table 1

#### SRMSEs
Get SRMSEs for intensity samples and my experiments:

```
clear; gensit summarise \
-dn DC/exp1 \
-et SIM_NN -et NonJointTableSIM_NN -et JointTableSIM_NN \
-el np -el MathUtils -el xr \
-e intensity_srmse_all_mean "intensity_srmse_all_by_seed.mean('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_all_std "intensity_srmse_all_by_seed.mean('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_train_mean "intensity_srmse_train_by_seed.mean('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_train_std "intensity_srmse_train_by_seed.std('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_test_mean "intensity_srmse_test_by_seed.mean('seed',dtype='float64',skipna=True)" \
-e intensity_srmse_test_std "intensity_srmse_test_by_seed.std('seed',dtype='float64',skipna=True)" \
-ea intensity \
-ea "test_cells=outputs.get_sample('test_cells')" \
-ea "train_cells=outputs.get_sample('train_cells')" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "srmse_func=MathUtils.srmse" \
-ea "intensity_mean=intensity.mean('iter',dtype='float64')" \
-ea "intensity_srmse_all_by_seed=srmse_func(prediction=intensity_mean,ground_truth=ground_truth)" \
-ea "intensity_srmse_train_by_seed=srmse_func(prediction=intensity_mean,ground_truth=ground_truth,mask=outputs.inputs.data.train_cells_mask)" \
-ea "intensity_srmse_test_by_seed=srmse_func(prediction=intensity_mean,ground_truth=ground_truth,mask=outputs.inputs.data.test_cells_mask)" \
-btt 'iter' 10000 10 100000 \
-k sigma -k type -k name -k title -fe intensity_SRMSEs -nw 20
```

Get SRMSEs for intensity samples and vanilla comparison experiments:

```
clear; gensit summarise -dn DC/comparisons \
-et XGBoost_Comparison -et GraphAttentionNetwork_Comparison -et GBRT_Comparison -et RandomForest_Comparison \
-el np -el MathUtils -el xr \
-e intensity_srmse_all_mean "intensity_srmse_all_mean_by_seed" \
-e intensity_srmse_all_std "intensity_srmse_all_std_by_seed" \
-e intensity_srmse_train_mean "intensity_srmse_train_mean_by_seed" \
-e intensity_srmse_train_std "intensity_srmse_train_std_by_seed" \
-e intensity_srmse_test_mean "intensity_srmse_test_mean_by_seed" \
-e intensity_srmse_test_std "intensity_srmse_test_std_by_seed" \
-ea intensity \
-ea "test_cells=outputs.get_sample('test_cells')" \
-ea "train_cells=outputs.get_sample('train_cells')" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "srmse_func=MathUtils.srmse" \
-ea "intensity_mean=intensity.mean('iter',dtype='float64')" \
-ea "intensity_srmse_all=srmse_func(prediction=intensity_mean,ground_truth=ground_truth)" \
-ea "intensity_srmse_train=srmse_func(prediction=intensity_mean,ground_truth=ground_truth,mask=outputs.inputs.data.train_cells_mask)" \
-ea "intensity_srmse_test=srmse_func(prediction=intensity_mean,ground_truth=ground_truth,mask=outputs.inputs.data.test_cells_mask)" \
-ea "intensity_srmse_all_mean_by_seed=intensity_srmse_all.mean('seed',dtype='float64',skipna=True)" \
-ea "intensity_srmse_all_mean_by_seed=intensity_srmse_all" \
-ea "intensity_srmse_train_mean_by_seed=intensity_srmse_train.mean('seed',dtype='float64',skipna=True)" \
-ea "intensity_srmse_train_mean_by_seed=intensity_srmse_train" \
-ea "intensity_srmse_test_mean_by_seed=intensity_srmse_test.mean('seed',dtype='float64',skipna=True)" \
-ea "intensity_srmse_test_mean_by_seed=intensity_srmse_test" \
-ea "intensity_srmse_all_std_by_seed=intensity_srmse_all.std('seed',dtype='float64',skipna=True)" \
-ea "intensity_srmse_all_std_by_seed=intensity_srmse_all" \
-ea "intensity_srmse_train_std_by_seed=intensity_srmse_train.std('seed',dtype='float64',skipna=True)" \
-ea "intensity_srmse_train_std_by_seed=intensity_srmse_train" \
-ea "intensity_srmse_test_std_by_seed=intensity_srmse_test.std('seed',dtype='float64',skipna=True)" \
-ea "intensity_srmse_test_std_by_seed=intensity_srmse_test" \
-btt 'iter' 10000 10 100000 \
-k type -k title -fe intensity_SRMSEs -nw 2
```

Get SRMSEs for table samples for my experiments:

clear; gensit summarise -dn DC/exp1 -et NonJointTableSIM_NN -et JointTableSIM_NN \
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
-ea "test_cells=outputs.get_sample('test_cells')" \
-ea "train_cells=outputs.get_sample('train_cells')" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "srmse_func=MathUtils.srmse" \
-ea "table_mean=table.mean('iter',dtype='float64')" \
-ea "table_srmse_all_by_seed=srmse_func(prediction=table_mean,ground_truth=ground_truth)" \
-ea "table_srmse_train_by_seed=srmse_func(prediction=table_mean,ground_truth=ground_truth,mask=outputs.inputs.data.train_cells_mask)" \
-ea "table_srmse_test_by_seed=srmse_func(prediction=table_mean,ground_truth=ground_truth,mask=outputs.inputs.data.test_cells_mask)" \
-cs "da.loss_name==str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])" \
-btt 'iter' 10000 9 1000000 \
-k sigma -k type -k name -k title -fe table_SRMSEs -nw 2
```

#### Coverage Probabilities

Get coverage probabilities for all samples and all experiments:

```
clear; gensit summarise \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-et SIM_NN -et SIM_MCMC -et JointTableSIM_NN -et JointTableSIM_MCMC -et NonJointTableSIM_NN  \ \
-el np -el MathUtils -el xr \
-e intensity_coverage "xr.apply_ufunc(roundint, 100*intensity_covered.mean(['origin','destination']))" \
-e table_coverage "xr.apply_ufunc(roundint, 100*table_covered.mean(['origin','destination']))" \
-ea table -ea intensity \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "coverage_func=MathUtils.coverage_probability" \
-ea "region_masses=[0.99]" \
-ea "roundint=MathUtils.roundint" \
-ea "intensity_covered=coverage_func(prediction=intensity,ground_truth=ground_truth,region_mass=region_masses)" \
-ea "table_covered=coverage_func(prediction=table,ground_truth=ground_truth,region_mass=region_masses)" \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])])" \
-k sigma -k type -k name -k title \
-fe CoverageProbabilities -nw 20
```

```
clear; gensit summarise \
-dn cambridge_work_commuter_lsoas_to_msoas/comparisons \
-et GBRT_Comparison -et GraphAttentionNetwork_Comparison -et RandomForest_Comparison -et XGBoost_Comparison \
-el np -el MathUtils -el xr \
-e intensity_coverage "xr.apply_ufunc(roundint, 100*intensity_covered.mean(['origin','destination']))" \
-ea intensity \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "coverage_func=MathUtils.coverage_probability" \
-ea "region_masses=[0.99]" \
-ea "roundint=MathUtils.roundint" \
-ea "intensity_covered=coverage_func(prediction=intensity,ground_truth=ground_truth,region_mass=region_masses)" \
-k type -k name -k title \
-fe CoverageProbabilities -nw 20
```
