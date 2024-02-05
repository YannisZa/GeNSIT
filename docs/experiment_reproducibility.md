<!-- # Cambridge commuter LSOAs to MSOAs -->

# Experiment runs

Set `ulimit -n 50000`

## Experiment 1

```
clear; multiresticodm run ./data/inputs/configs/experiment1_disjoint.toml -sm -et NonJointTableSIM_NN -nt 4 -nw 7
clear; multiresticodm run ./data/inputs/configs/experiment1_joint.toml -sm -et JointTableSIM_NN -nt 4 -nw 7
```

## Experiment 2

```
clear; multiresticodm run ./data/inputs/configs/experiment2_disjoint.toml -sm -et NonJointTableSIM_NN -nt 20 -nw 1
clear; multiresticodm run ./data/inputs/configs/experiment2_joint.toml -sm -et JointTableSIM_NN -nt 20 -nw 1
```

## Experiment 3

```
clear; multiresticodm run ./data/inputs/configs/experiment3_joint.toml -sm -et JointTableSIM_NN -nt 5 -nw 7
```

# Summaries and Metrics

## Table 1

Get SRMSEs for all samples and all experiments:

```
clear; multiresticodm summarise \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-et SIM_NN -et SIM_MCMC -et JointTableSIM_NN -et JointTableSIM_MCMC -et NonJointTableSIM_NN  \
-el np -el MathUtils -el xr \
-e table_srmse "srmse_func(prediction=mean_table,ground_truth=ground_truth)" \
-e intensity_srmse "srmse_func(prediction=mean_intensity,ground_truth=ground_truth)" \
-ea table -ea intensity -ea sign \
-ea "srmse_func=MathUtils.srmse" \
-ea "signed_mean_func=outputs.compute_statistic" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "mean_table=table.mean(['id'])" \
-ea "mean_intensity=signed_mean_func(intensity,'intensity','signedmean',dim=['id'])" \
-ea "mean_intensity=intensity.mean(['id'])" \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss']),str(['table_likelihood_loss'])])" \
-btt 'iter' 10000 90 1000 \
-k sigma -k type -k name -k title -fe SRMSEs -nw 20
```

Get coverage probabilities for all samples and all experiments:

```
clear; multiresticodm summarise \
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

### Experiment-specific summaries

Get SRMSEs for all samples:

```
clear; multiresticodm summarise \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-d NonJointTableSIM_NN_SweepedNoise_01_02_2024_16_51_58  \
-el np -el MathUtils -el xr \
-e table_srmse "srmse_func(prediction=mean_table,ground_truth=ground_truth)" \
-e intensity_srmse "srmse_func(prediction=mean_intensity,ground_truth=ground_truth)" \
-e table_size "dict(table.sizes)" \
-e intensity_size "dict(intensity.sizes)" \
-ea table -ea intensity \
-ea "srmse_func=MathUtils.srmse" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "mean_table=table.mean(['id'])" \
-ea "mean_intensity=intensity.mean(['id'])" \
-btt 'iter' 100 100 1000 \
-k sigma -k type -k name -k title -fe SRMSEs -nw 20
```

Get coverage probabilities for all samples:

```
clear; multiresticodm summarise \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-d NonJointTableSIM_NN_SweepedNoise_01_02_2024_16_51_58 \
-el np -el MathUtils -el MiscUtils -el xr \
-e table_coverage "xr.apply_ufunc(roundint, 100*table_covered.mean(['origin','destination']))" \
-e intensity_coverage "xr.apply_ufunc(roundint, 100*intensity_covered.mean(['origin','destination']))" \
-ea table -ea intensity \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "coverage_probability=MathUtils.coverage_probability" \
-ea "region_mass=[0.99]" \
-ea "roundint=MathUtils.roundint" \
-ea "apply_and_combine=MiscUtils.xr_apply_and_combine_wrapper" \
-ea "functions=[{'coverage_probability':{'callable':coverage_probability,'apply_ufunc':False}}]" \
-ea "isolated_sweeped_kwargs={'region_mass':region_mass}" \
-ea "fixed_kwargs={'coverage_probability':{'ground_truth':ground_truth}}" \
-ea "table_covered=apply_and_combine(table,functions=functions,fixed_kwargs=fixed_kwargs,isolated_sweeped_kwargs=isolated_sweeped_kwargs)" \
-ea "intensity_covered=apply_and_combine(intensity,functions=functions,fixed_kwargs=fixed_kwargs,isolated_sweeped_kwargs=isolated_sweeped_kwargs)" \
-k sigma -k type -k name -k title \
-fe CoverageProbabilities -nw 20
```

# Plots

## Figure 1

```
clear; multiresticodm plot 2d line --y_shade --y_group 'type' -y table_density -x density_eval_points \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 -et NonJointTableSIM_NN -et JointTableSIM_NN -et JointTableSIM_MCMC \
-el np -el ProbabilityUtils -el xr \
-e table_density "xr.apply_ufunc(kernel_density,table_like_loss.groupby('sweep'),kwargs={'x':xs},exclude_dims=set(['id']),input_core_dims=[['id']],output_core_dims=[['id']])" \
-e density_eval_points "xr.DataArray(xs)" \
-e table_density_height "np.nanmax(table_density)" \
-ea "xs=np.linspace(2.8,4.2,1000)" \
-ea table -ea intensity \
-ea "kernel_density=ProbabilityUtils.kernel_density" \
-ea "table_lossfn=outputs.ct_mcmc.table_loss" \
-ea "table_like_loss=table_lossfn(table/table.sum(['origin','destination']),np.log(intensity/intensity.sum(['origin','destination'])))" \
-btt 'iter' 100 100 1000 \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])])" \
-cs "~da.title.isin(['_unconstrained','_total_constrained','_total_intensity_row_table_constrained'])" \
-hchv 0.5 -hch sigma -c title -op 1.0 -msz 1 -l title -l sigma -or asc table_density_height -k sigma -k type -k title \
-ft 'figure1_table_like_loss_kernel_density' -xlab 'Table loss' \
-fs 6 10 -lls 8 -sals 12 -salr 90 -salp 2 -tls 7 -xlim 2.8 3.7 -la 0 0 -nw 20
```

Load plot data and replot

```
clear; multiresticodm plot 2d line --y_shade \
-pdd ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp1/paper_figures/ \
-ft 'figure1_table_like_loss_kernel_density' -xlab '$L\left(\mathbf{T},\boldsymbol{\Lambda}\right)$' \
-fs 6 10 -lls 9 -sals 12 -salr 90 -tls 10 -xtr 0 0 -ytr 0 0 -xtp 3 3 -ytp 3 3 \
-xlim 3.6 3.7 -xlim 2.8 3.7 -xlim 3.6 3.7 -la 1 0 -loc 'upper left' -nw 20
```

## Figure 2

Plot cumulative SRMSEs and CPs for every constraint and sampling method. Do this for the tables samples:

```
clear; multiresticodm plot 2d scatter -y table_srmse -x type -x end --x_discrete \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-et JointTableSIM_MCMC -et JointTableSIM_NN -et NonJointTableSIM_NN \
-el np -el MathUtils -el MiscUtils -el xr \
-e table_coverage_probability "xr.apply_ufunc(roundint, 100*table_coverage.mean(['origin','destination'])).astype('int32')" \
-e table_coverage_probability_size "xr.apply_ufunc(lambda x: np.exp(6*x), table_coverage.mean(['origin','destination']))" \
-e table_srmse "apply_and_combine(table,functions=srmse_functions,fixed_kwargs=fixed_kwargs,isolated_sweeped_kwargs=isolated_sweeped_kwargs)" \
-ea table \
-ea "endings=[10000,20000,40000,60000,80000,100000]" -ea "region_mass=[0.99]" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "apply_and_combine=MiscUtils.xr_apply_and_combine_wrapper" -ea "srmse=MathUtils.srmse" -ea "islice=MiscUtils.xr_islice" -ea "coverage_probability=MathUtils.coverage_probability" -ea "sample_mean=MathUtils.sample_mean" -ea "roundint=MathUtils.roundint" \
-ea "cp_functions=[{'islice':{'callable':islice}},{'coverage_probability':{'callable':coverage_probability}}]" \
-ea "srmse_functions=[{'islice':{'callable':islice}},{'sample_mean':{'callable':sample_mean}},{'srmse':{'callable':srmse}}]" \
-ea "fixed_kwargs={'islice':{'dim':'id'},'coverage_probability':{'ground_truth':ground_truth},'srmse':{'ground_truth':ground_truth},'sample_mean':{'dim':[str('id')]}}" \
-ea "isolated_sweeped_kwargs={'end':endings,'region_mass':region_mass}" \
-ea "table_coverage=apply_and_combine(table,functions=cp_functions,fixed_kwargs=fixed_kwargs,isolated_sweeped_kwargs=isolated_sweeped_kwargs)" \
-k sigma -k type -k name -k title \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])])" \
-cs "~da.title.isin(['_unconstrained','_total_constrained','_total_intensity_row_table_constrained'])" \
-c title -op 1.0 -mrkr sigma -msz table_coverage_probability_size -an table_coverage_probability -l title -l sigma \
-ft 'figure2/cumulative_srmse_and_cp_by_method' -ylab 'SRMSE$(\mathbb{E}\[\mytable^{(1:N)}\],\mytable^{\mathcal{D}})$' -xlab '(Method,$N$)' \
-fs 10 10 -la 0 0 -lc 2 -loc 'upper right' -bbta 1.0 1.0 -lls 16 -ylr 90 -xls 20 -yls 20 -yts 18 18 -xts 18 18 \
-xtr 45 0 -xtp 0 100 -ytl 0.2 -xtl 1 1 -xtl 2 3 -xlim 0 19 -ylim 0 2.1 -nw 20
```

Load plot data and replot

```
clear; multiresticodm plot 2d scatter -y table_srmse -x type -x end --x_discrete \
-pdd ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp1/paper_figures/figure2/ \
-ft 'cumulative_srmse_and_cp_by_method' -ylab 'SRMSE$(\mathbb{E}[\mytable^{(1:N)}],\mytable^{\mathcal{D}})$' -xlab '(Method,$N$)' \
-fs 10 10 -la 0 0 -lc 2 -loc 'upper right' -bbta 1.0 1.0 -lls 16 -ylr 90 -xls 20 -yls 20 -yts 18 18 -xts 18 18 \
-xtr 75 0 -xtp 0 100 -ytl 0.2 -xtl 1 1 -xtl 2 3 -xlim 0 19 -ylim 0 2.1 -nw 20
```

DITTO for the intensity samples:

## Figure 3

-et JointTableSIM_NN

```
clear; multiresticodm plot 2d scatter -y srmse -x type -x 'iter&seed' --x_discrete  \
-dn cambridge_work_commuter_lsoas_to_msoas/exp2 -et NonJointTableSIM_NN_SweepedNoise_01_02_2024_14_38_56 \
-el np -el MathUtils -el MiscUtils -el xr \
-e table_coverage_probability_size "xr.apply_ufunc(lambda x: np.exp(6*x), table_coverage.mean(['origin','destination'])).unstack('id')" \
-e table_srmse "srmse(prediction=mean_table,ground_truth=ground_truth).unstack('id')" \
-ea table \
-ea "mean_table=table.mean(['id'])" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "srmse=MathUtils.srmse" -ea "coverage_probability=MathUtils.coverage_probability" -ea "roundint=MathUtils.roundint" \
-ea "table_coverage=coverage_probability(prediction=table,ground_truth=ground_truth)" \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss']),str(['table_likelihood_loss'])])" \
-k sigma -k type -k name -k title -k seed -k iter \
-mrkr sigma -c title -msz table_coverage_probability_size -op 0.5 -or asc table_coverage_probability_size -l sigma -l title \
-fs 10 10 -ft 'figure3/exploration_exploitation_tradeoff_srmse_cp_vs_method_epoch_seed' \
-xlab 'Method, ($N$, Ensemble size)' -ylab 'SRMSE$(\mathbb{E}\[\mytable^{(1:N)}\],\mytable^{\mathcal{D}})$' \
-ylim 0.0 2.2 -xtr 0 0 -xtp 0 100 -ytl 0.2 -xtl 1 1 -xtl 1 2 -lls 8 -xts 8 8 -xts 8 8 -nw 20
```

```
-e intensity_srmse "srmse_func(prediction=mean_intensity,ground_truth=ground_truth)" \
-ea intensity -ea sign \
-ea "signed_mean_func=outputs.compute_statistic" \
-ea "mean_intensity=signed_mean_func(intensity,'intensity','signedmean',dim=['id'])" \
-ea "mean_intensity=intensity.mean(['id'])" \
```

Load plot data and replot

```
clear; multiresticodm plot 2d scatter -y srmse -x type -x 'iter&seed' --x_discrete  \
-pdd ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp1/paper_figures/figure3/ \
-fs 10 10 -ft 'figure3/exploration_exploitation_tradeoff_srmse_cp_vs_method_epoch_seed' \
-xlab 'Method, ($N$, Ensemble size)' -ylab 'SRMSE$(\mathbb{E}\[\mytable^{(1:N)}\],\mytable^{\mathcal{D}})$' \
-ylim 0.0 2.2 -xtr 0 0 -xtp 0 100 -ytl 0.2 -xtl 1 1 -xtl 1 2 -lls 8 -xts 8 8 -xts 8 8 -nw 20
```

## Figure 4

```
clear; multiresticodm plot 2d scatter -y srmse -x type -x 'iter&seed' --x_discrete  \
-dn cambridge_work_commuter_lsoas_to_msoas/exp3 -et JointTableSIM_NN_SweepedNoise_01_02_2024_14_55_23 \
-el np -el MathUtils -el MiscUtils -el xr \
-e table_coverage_probability_size "xr.apply_ufunc(lambda x: np.exp(6*x), table_coverage.mean(['origin','destination'])).unstack('id')" \
-e table_srmse "srmse(prediction=mean_table,ground_truth=ground_truth).unstack('id')" \
-ea table \
-ea "mean_table=table.mean(['id'])" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "srmse=MathUtils.srmse" -ea "coverage_probability=MathUtils.coverage_probability" -ea "roundint=MathUtils.roundint" \
-ea "table_coverage=coverage_probability(prediction=table,ground_truth=ground_truth)" \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss']),str(['table_likelihood_loss'])])" \
-k sigma -k type -k name -k title -k seed -k iter \
-mrkr sigma -c title -op 0.5 -or asc table_coverage_probability_size -msz 'table_coverage_probability_size' -l sigma -l title \
-fs 10 10 -ft 'figure3/exploration_exploitation_tradeoff_srmse_cp_vs_method_epoch_seed' \
-xlab 'Method, ($N$, Ensemble size)' -ylab 'SRMSE$(\mathbb{E}\[\mytable^{(1:N)}\],\mytable^{\mathcal{D}})$' \
-ylim 0.0 2.2 -xtr 0 0 -xtp 0 100 -ytl 0.0 0.2 -xtl 1 1 -xtl 1.5 2 -lls 8 -xts 8 8 -xts 8 8 -nw 20
```

# VARIOUS TESTS
