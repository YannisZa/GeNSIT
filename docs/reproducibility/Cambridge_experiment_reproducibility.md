# Cambridge work commuter dataset

Set `ulimit -n 50000`

## Experiment 1

```
clear; gensit run ./data/inputs/configs/cambridge/experiment1_disjoint.toml -sm -et NonJointTableSIM_NN -nt 4 -nw 7
clear; gensit run ./data/inputs/configs/cambridge/experiment1_joint.toml -sm -et JointTableSIM_NN -nt 4 -nw 7
clear; gensit run ./data/inputs/configs/cambridge/repeated_experiments.toml -sm -et JointTableSIM_MCMC -nt 4 -nw 7
```

## Experiment 2

```
clear; gensit run ./data/inputs/configs/cambridge/experiment2_disjoint.toml -sm -et NonJointTableSIM_NN -nt 20 -nw 1
clear; gensit run ./data/inputs/configs/cambridge/experiment2_joint.toml -sm -et JointTableSIM_NN -nt 20 -nw 1
```

## Experiment 3

```
clear; gensit run ./data/inputs/configs/cambridge/experiment3_joint.toml -sm -et JointTableSIM_NN -nt 5 -nw 7
```

# Summaries and Metrics

## Table 1

Get SRMSEs for all samples and all experiments:

```
clear; gensit summarise \
-dn cambridge/exp1 \
-et SIM_NN -et SIM_MCMC -et JointTableSIM_NN -et JointTableSIM_MCMC -et NonJointTableSIM_NN  \
-el np -el MathUtils -el xr \
-e table_srmse "srmse_func(prediction=mean_table,ground_truth=ground_truth)" \
-e intensity_srmse "srmse_func(prediction=mean_intensity,ground_truth=ground_truth)" \
-ea table -ea intensity -ea sign \
-ea "srmse_func=MathUtils.srmse" \
-ea "signed_mean_func=MathUtils.signed_mean" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "mean_table=table.mean(['id'])" \
-ea "mean_intensity=signed_mean_func(intensity,'intensity','signedmean',dim=['id'])" \
-ea "mean_intensity=intensity.mean(['id'])" \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss']),str(['table_likelihood_loss'])])" \
--slice -btt 'iter' 10000 90 1000 \
-k sigma -k type -k name -k title -fe SRMSEs -nw 20
```

Get coverage probabilities for all samples and all experiments:

```
clear; gensit summarise \
-dn cambridge/exp1 \
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
--slice -k sigma -k type -k name -k title \
-fe CoverageProbabilities -nw 20
```

### Experiment-specific summaries

Get SRMSEs for all samples:

```
clear; gensit summarise \
-dn cambridge/exp1 \
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

```
clear; gensit summarise -dn cambridge/comparisons \
-et GraphAttentionNetwork_Comparison \
-el np -el MathUtils -el xr \
-e "intensity_all_srmse" "MathUtils.srmse(intensity_mean,ground_truth=ground_truth_table,mask=test_cells_mask)" \
-e "intensity_train_srmse" "MathUtils.srmse(intensity_mean,ground_truth=ground_truth_table,mask=train_cells_mask)" \
-ea intensity \
-ea "intensity_mean=intensity.mean('iter',dtype='float64')" \
-btt 'iter' 10000 9 10000 \
-k type -k title -fe intensity_SRMSEs -nw 2
```

Get SSI for all samples:
```
clear; gensit summarise -dn cambridge/comparisons \
-et GraphAttentionNetwork_Comparison \
-el np -el MathUtils -el xr \
-e intensity_ssi_train "MathUtils.ssi(intensity_mean,ground_truth=ground_truth_table,mask=train_cells_mask)" \
-e intensity_ssi_test "MathUtils.ssi(intensity_mean,ground_truth=ground_truth_table,mask=test_cells_mask)" \
-ea intensity \
-ea "intensity_mean=intensity.mean('iter',dtype='float64')" \
-btt 'iter' 10000 9 10000 \
-k type -k title -fe intensity_SSIs -nw 2
```


Get coverage probabilities for all samples:

```
clear; gensit summarise \
-dn cambridge/exp1 \
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

```
clear; gensit summarise  -dn cambridge/comparisons \
-et GraphAttentionNetwork_Comparison \
-el np -el MathUtils -el xr \
-e intensity_cp_train "xr.apply_ufunc(roundint, 100*intensity_cp_train.mean(['origin','destination'],skipna=True))" \
-e intensity_cp_test "xr.apply_ufunc(roundint, 100*intensity_cp_test.mean(['origin','destination'],skipna=True))" \
-ea intensity \
-ea "cp_func=MathUtils.coverage_probability" \
-ea "roundint=MathUtils.roundint" \
-ea "region_masses=0.99" \
-ea "intensity_cp_train=cp_func(intensity.stack(id=['iter']),ground_truth=ground_truth_table,mask=train_cells_mask,region_mass=region_masses)" \
-ea "intensity_cp_test=cp_func(intensity.stack(id=['iter']),ground_truth=ground_truth_table,region_mass=region_masses,mask=test_cells_mask)" \
-btt 'iter' 10000 9 10000 \
-k type -k title -fe intensity_CoverageProbabilities -nw 2
```


# Plots

## Figure 1

```
clear; gensit plot simple line --y_shade --y_group 'type' -y table_density -x density_eval_points \
-dn cambridge/exp1 -et JointTableSIM_MCMC -et NonJointTableSIM_NN -et JointTableSIM_NN \
-el np -el ProbabilityUtils -el xr \
-e table_density "xr.apply_ufunc(kernel_density,table_like_loss.groupby('sweep'),kwargs={'x':xs,'bandwidth':bandwidth},exclude_dims=set(['id']),input_core_dims=[['id']],output_core_dims=[['id']])" \
-e density_eval_points "xr.DataArray(xs)" \
-e table_density_height "np.nanmax(table_density)" \
-fa "{'xs':np.linspace(3.6,3.7,1000),'bandwidth':0.25}" \
-fa "{'xs':np.linspace(3.6,3.7,1000),'bandwidth':0.25}" \
-fa "{'xs':np.linspace(2.4,4.2,1000),'bandwidth':0.25}" \
-ea table -ea intensity \
-ea "kernel_density=ProbabilityUtils.kernel_density" \
-ea "table_lossfn=outputs.ct_mcmc.table_loss" \
-ea "table_like_loss=table_lossfn(table/table.sum(['origin','destination']),np.log(intensity/intensity.sum(['origin','destination'])))" \
-btt 'iter' 100 100 1000 \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])])" \
-cs "~da.title.isin(['_unconstrained','_total_intensity_row_table_constrained'])" \
--slice -k sigma -k type -k title \
-ft 'figure1/figure1_table_like_loss_kernel_density' -ff ps \
-hchv 0.5 -hch sigma -c title -op 1.0 -msz 1 -l title -l sigma -or asc table_density_height \
-xlab '$\lossoperator(\mytable,\myintensity)$' -ylab 'Kernel density' \
-fs 6 10 -lls 11 -lp 0.5 -la 2 0 -lc 2 -loc 'lower center' -bbta 0.5 -bbta -1.1  \
-ylr 90 -xts 12 12 -yts 12 12 -xtp 0 0 -ytp 0 0 -ylp 2 -xlp 1 \
-xlim 2.8 3.7 -xlim 3.61 3.67 -xlim 3.61 3.67 -ylim 0 10 -ylim 0 300 -ylim 0 500 -hlw 0.2
```

Load plot data and replot

```
clear; gensit plot simple line --y_shade --y_group 'type' -y table_density -x density_eval_points \
-pdd ./data/outputs/cambridge/exp1/paper_figures/figure1/ \
-ft 'figure1_table_like_loss_kernel_density' -ff ps \
-xlab '$\lossoperator (\mytable,\myintensity)$' -ylab 'Kernel density' \
-fs 6 10 -lls 11 -lp 0.5 -la 2 0 -lc 2 -loc 'lower center' -bbta 0.5 -bbta -1.1  \
-ylr 90 -xts 12 12 -yts 12 12 -xtp 0 0 -ytp 0 0 -ylp 2 -xlp 1 \
-xlim 2.8 3.7 -xlim 3.61 3.67 -xlim 3.61 3.67 -ylim 0 10 -ylim 0 300 -ylim 0 500 -hlw 0.2
```

-ylim 0 50 -ylim 0 500 -ylim 0 500

## Figure 2

Plot cumulative SRMSEs and CPs for every constraint and sampling method. Do this for the tables samples:

```
clear; gensit plot simple scatter -y table_srmse -x type -x end --x_discrete \
-dn cambridge/exp1 -o ./data/outputs/ \
-et JointTableSIM_MCMC -et JointTableSIM_NN -et NonJointTableSIM_NN \
-el np -el MathUtils -el MiscUtils -el xr \
-e table_coverage_probability "xr.apply_ufunc(roundint, 100*table_coverage.mean(['origin','destination'])).astype('int32')" \
-e table_coverage_probability_size "xr.apply_ufunc(lambda x: np.exp(8*x-2), table_coverage.mean(['origin','destination']))" \
-e table_srmse "apply_and_combine(table,functions=srmse_functions,fixed_kwargs=fixed_kwargs,isolated_sweeped_kwargs=isolated_sweeped_kwargs)" \
-ea table \
-ea "endings=[10000,20000,40000,60000,80000,100000]" \
-ea "region_mass=[0.99]" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "apply_and_combine=MiscUtils.xr_apply_and_combine_wrapper" \
-ea "srmse=MathUtils.srmse" \
-ea "islice=MiscUtils.xr_islice" \
-ea "coverage_probability=MathUtils.coverage_probability" \
-ea "sample_mean=MathUtils.sample_mean" \
-ea "roundint=MathUtils.roundint" \
-ea "cp_functions=[{'islice':{'callable':islice}},{'coverage_probability':{'callable':coverage_probability}}]" \
-ea "srmse_functions=[{'islice':{'callable':islice}},{'sample_mean':{'callable':sample_mean}},{'srmse':{'callable':srmse}}]" \
-ea "fixed_kwargs={'islice':{'dim':'iter'},'coverage_probability':{'ground_truth':ground_truth,'dim':'iter'},'srmse':{'ground_truth':ground_truth},'sample_mean':{'dim':'iter'}}" \
-ea "isolated_sweeped_kwargs={'end':endings,'region_mass':region_mass}" \
-ea "table_coverage=apply_and_combine(table,functions=cp_functions,fixed_kwargs=fixed_kwargs,isolated_sweeped_kwargs=isolated_sweeped_kwargs)" \
-k sigma -k type -k name -k title \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])])" \
-cs "~da.title.isin(['_unconstrained'])" \
--slice -c title -op 1.0 -mrkr sigma -msz table_coverage_probability_size -l title -l sigma -or asc table_coverage_probability_size  \
-ft 'figure2/cumulative_srmse_and_cp_by_method' -ylab 'SRMSE' -xlab 'Method, $N$' \
-la 0 0 -lc 2 -loc 'upper center' -bbta 0.5 -bbta 1.35 -lls 14 -ylr 90 -xls 20 -yls 20 -yts 18 18 -xts 12 16 \
-xtp 0 102 -ytl 0.0 0.2 -xtl 1 1 -xtl 2 3 -xlim 0 19 -ylim 0 1.8 -xtr 75 0
```

Load plot data and replot

```
clear; gensit plot simple scatter -y table_srmse -x type -x end --x_discrete \
-pdd ./data/outputs/cambridge/exp1/paper_figures/figure2/ \
-fs 10 10 -ff ps -ft 'cumulative_srmse_and_cp_by_method' \
-ylab 'SRMSE' -xlab 'Method, $N$' \
-la 0 0 -lc 2 -loc 'upper center' -bbta 0.5 -bbta 1.35 -lls 14 -ylr 90 -xls 20 -yls 20 -yts 18 18 -xts 12 16 \
-xtp 0 102 -ytl 0.0 0.2 -xtl 1 1 -xtl 2 3 -xlim 0 19 -ylim 0 1.8 -xtr 75 0
```

## Figure 3

```
pkill -9 -f "gensit plot simple scatter -y table_srmse -x type -x 'N&ensemble_size'"; \
clear; gensit plot simple scatter -y table_srmse -x type -x 'N&ensemble_size' --x_discrete -gb seed  \
-dn cambridge/exp2 -o ./data/outputs/ \
-et NonJointTableSIM_NN -et JointTableSIM_NN \
-el np -el MathUtils -el MiscUtils -el xr \
-e table_coverage_probability_size "table_coverage.mean(['origin','destination'])" \
-e table_srmse "srmse(prediction=mean_table,ground_truth=ground_truth)" \
-e ensemble_size "table_srmse.copy(data=[len(table.coords['seed'].values)])" \
-ea table \
-ea "mean_table=table.mean(['iter','seed'])" \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "srmse=MathUtils.srmse" \
-ea "coverage_probability=MathUtils.coverage_probability" \
-ea "roundint=MathUtils.roundint" \
-ea "table_coverage=coverage_probability(prediction=table.stack(id=['seed','iter']),ground_truth=ground_truth,dim='id')" \
-k sigma -k type -k name -k title -k N \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])])" \
-cs "~da.title.isin(['_unconstrained','_total_intensity_row_table_constrained'])" \
--slice -mrkr sigma -c title -msz table_coverage_probability_size -op 1.0 -or asc table_coverage_probability_size -l sigma -l title \
-fs 10 10 -ff ps -ft 'figure3_rerun/exploration_exploitation_tradeoff_srmse_cp_vs_method_epoch_seed' \
-xlab 'Method, ($N$, $E$)' -ylab 'SRMSE' \
-ylim 0.0 3.2 -ylr 90 -xtp 0 80 -ytl 0.0 0.2 -ytl 0.0 0.0 -xtl 5 8 -xtl 9 16 -yts 18 18 -xts 18 18 -xts 18 18 \
-xtr 70 0 -xls 20 -yls 20 -xlim 0 111 -la 0 0 -lls 14 -loc 'upper center' -bbta 0.45 -bbta 1.3 -btta 0.4 -btta 1.0 -lc 3 -lp 0.01 -lcs 0.1
```

Load plot data and replot

```
clear; gensit plot simple scatter -y table_srmse -x type -x 'N&ensemble_size' --x_discrete -gb seed  \
-pdd ./data/outputs/cambridge/exp2/paper_figures/figure3/ \
-fs 10 10 -ff ps -ft 'exploration_exploitation_tradeoff_srmse_cp_vs_method_epoch_seed' \
-xlab 'Method, ($N$, $E$)' -ylab 'SRMSE' \
-ylim 0.0 3.2 -ylr 90 -xtp 0 80 -ytl 0.0 0.2 -ytl 0.0 0.0 -xtl 5 8 -xtl 9 16 -yts 18 18 -xts 18 18 -xts 18 18 \
-xtr 70 0 -xls 20 -yls 20 -xlim 0 111 -la 0 0 -lls 14 -loc 'upper center' -bbta 0.45 -bbta 1.3 -btta 0.4 -btta 1.0 -lc 3 -lp 0.01 -lcs 0.1
```

## Figure 4

```
clear; gensit plot simple scatter -y table_srmse -x loss_name --x_discrete  \
-dn cambridge/exp3 -o ./data/outputs/ -et JointTableSIM_NN \
-el np -el MathUtils -el xr \
-e table_coverage_probability_size "coverage_probability(prediction=table,ground_truth=ground_truth,dim='iter').mean(['origin','destination'])" \
-e table_srmse "srmse(prediction=table.mean('iter'),ground_truth=ground_truth)" \
-ea table \
-ea "ground_truth=outputs.inputs.data.ground_truth_table" \
-ea "srmse=MathUtils.srmse" \
-ea "coverage_probability=MathUtils.coverage_probability" \
-k sigma -k type -k name -k title -k loss_name \
-cs "~da.loss_name.isin([str(['total_table_distance_loss','total_intensity_distance_loss'])])" \
-cs "~da.title.isin(['_unconstrained','_total_intensity_row_table_constrained'])" \
--slice -fs 10 10 -ff ps -ft 'figure4_rerun/loss_function_validation_intractable_odms' \
-mrkr sigma -c title -msz table_coverage_probability_size -or asc table_coverage_probability_size -l sigma -l title \
-xlab 'Loss operator $\lossoperator$' -ylab 'SRMSE' \
-ylim 0.0 2.2 -xtr 0 0 -xtp 0 100 -ytl 0.0 0.2 -xtl 1 1 -xtl 1.5 2 -lls 8 -xts 8 8 -xts 8 8 -nw 1
```

-cs "da.title.isin(['_total_constrained','_row_constrained'])" \
-fs 10 10 -ff ps -ft 'figure4/loss_function_validation_tractable_odms' \

-cs "da.title.isin(['_total_constrained','_row_constrained','_doubly_constrained','_doubly_10%_cell_constrained','_doubly_20%_cell_constrained'])" \
-fs 10 10 -ff ps -ft 'figure4/loss_function_validation_all_odms' \
Load plot data and replot

```
clear; gensit plot simple scatter -y table_srmse -x loss_name --x_discrete  \
-pdd ./data/outputs/cambridge/exp3/paper_figures/figure4/ \
-fs 10 10 -ff ps -ft 'figure4_loss_function_validation_all_odms' \
-xlab 'Loss operator $\lossoperator$' -ylab 'SRMSE' \
-la 0 0 -lc 2 -loc 'upper center' -bbta 0.5 -bbta 1.4 -lls 14 -ylr 90 -xls 20 -yls 20 -yts 18 18 -xts 12 16 \
-xtp 0 92 -ytl 0.0 0.2 -xtl 3 3 -xtl 3 3 -xlim 0 27 -ylim 0 1.4 -xtr 75 0
```