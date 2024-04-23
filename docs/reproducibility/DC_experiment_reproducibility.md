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
clear; gensit run ./data/inputs/configs/DC/experiment1_nn_disjoint.toml -et SIM_NN -nt 6 -nw 5
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

## Summaries and Metrics

### Table 1

Get SRMSEs for all samples and all experiments:

```
clear; gensit summarise \
-dn DC/exp1 \
-et SIM_NN -et NonJointTableSIM_NN -et JointTableSIM_NN \
-el np -el MathUtils -el xr \
-e table_srmse_all "srmse_func(prediction=mean_table,ground_truth=ground_truth)" \
-e intensity_srmse_all "srmse_func(prediction=mean_intensity,ground_truth=ground_truth)" \
-e table_srmse_train "srmse_func(prediction=mean_table,ground_truth=ground_truth,cells=train_cells)" \
-e intensity_srmse_train "srmse_func(prediction=mean_intensity,ground_truth=ground_truth,cells=train_cells)" \
-e table_srmse_test "srmse_func(prediction=mean_table,ground_truth=ground_truth,cells=test_cells)" \
-e intensity_srmse_test "srmse_func(prediction=mean_intensity,ground_truth=ground_truth,cells=test_cells)" \
-ea table -ea intensity \
-ea "test_cells=outputs.get_sample('test_cells')" \
-ea "train_cells=outputs.get_sample('train_cells')" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "srmse_func=MathUtils.srmse" \
-ea "mean_table=table.mean(['id'],dtype='float64')" \
-ea "mean_intensity=intensity.mean(['id'],dtype='float64')" \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss']),str(['table_likelihood_loss'])])" \
-btt 'iter' 10000 90 1000 \
-vd test_cells "./data/inputs/DC/test_cells.txt" -vd train_cells "./data/inputs/DC/train_cells.txt" \
-k sigma -k type -k name -k title -fe SRMSEs -nw 20
```

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
