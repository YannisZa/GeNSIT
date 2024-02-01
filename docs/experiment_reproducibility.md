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
clear; multiresticodm run ./data/inputs/configs/experiment3_joint.toml -sm -et JointTableSIM_NN -nt 6 -nw 5
```

# Summaries and Metrics

## Table 1

Get SRMSEs for all samples and all experiments
-et SIM_NN -et SIM_MCMC -et JointTableSIM_NN -et JointTableSIM_MCMC -et NonJointTableSIM_NN

```
clear; multiresticodm summarise \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-et NonJointTableSIM_NN  \
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

```
clear; multiresticodm summarise \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-d NonJointTableSIM_NN_SweepedNoise_01_02_2024_00_45_29  \
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
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss']),str(['table_likelihood_loss'])])" \
-k sigma -k type -k name -k title -fe SRMSEs -nw 20
```

Get coverage probabilities for all samples and all experiments:

```
clear; multiresticodm summarise \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-et SIM_NN -et SIM_MCMC \
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

# Plots

## Figure 1

```
clear; multiresticodm plot 2d line --y_shade --y_group 'type' -y table_density -x density_eval_points \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 -et NonJointTableSIM_NN -et JointTableSIM_NN -et JointTableSIM_MCMC \
-el np -el ProbabilityUtils -el xr \
-e table_density "xr.apply_ufunc(kernel_density,mean_table_like_loss.groupby('sweep'),kwargs={'x':xs},exclude_dims=set(['iter']),input_core_dims=[['iter']],output_core_dims=[['iter']])" \
-e density_eval_points "xr.DataArray(xs)" \
-e table_density_height "np.nanmax(table_density)" \
-ea "xs=np.linspace(3.0,4.0,1000)" \
-ea table -ea intensity \
-ea "kernel_density=ProbabilityUtils.kernel_density" \
-ea "table_lossfn=outputs.ct_mcmc.table_loss" \
-ea "table_like_loss=table_lossfn(table/table.sum(['origin','destination']),np.log(intensity/intensity.sum(['origin','destination'])))" \
-ea "mean_table_like_loss=table_like_loss.rename(iter='iter_old').stack(iter=['iter_old','seed'])" \
-ea "mean_table_like_loss=table_like_loss.set_index(iter='iter')" \
-btt 'iter' 10000 100 100 \
-btt 'iter' 0 75 13 \
-cs "da.loss_name.isin([str(['dest_attraction_ts_likelihood_loss']),str(['dest_attraction_ts_likelihood_loss', 'table_likelihood_loss'])])" \
-cs "~da.title.isin(['_unconstrained','_total_constrained'])" \
-hch sigma -c title -v 1.0 -sz 1 -l title -l sigma -or asc table_density_height -k sigma -k type -k title \
-ft 'table_like_loss_kernel_density' -xlab 'Table loss' \
-fs 6 10 -lls 89 -sals 12 -salr 90 -salp 2 -tls 7 -nw 20 -xlim 3.5 4.0 -la 0 0
```

Load plot data and replot

```
clear; multiresticodm plot 2d line --y_shade \
-pdd ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp1/paper_figures/ \
-ft 'table_like_loss_kernel_density' -xlab 'Table loss' \
-fs 6 10 -lls 9 -sals 12 -salr 90 -salp 2 -tls 7 -nw 20 -xlim 3.5 4.0 -la 0 0
```

## Figure 2

## Figure 3

`pkill -9 -f 'multiresticodm plot'; `

-pdd /home/iz230/MultiResTICODM/data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp2/NonJointTableSIM_NN_SweepedNoise\_\_31_10_2023_09_44_49/paper_figures/ -dn cambridge_work_commuter_lsoas_to_msoas/exp2 \

```

-et JointTableSIM_NN

clear; multiresticodm plot -y srmse -x 'iter&seed' -x sigma \
-dn cambridge_work_commuter_lsoas_to_msoas/exp2 -et NonJointTableSIM_NN \
-ma table -ma intensity -ft 'srmse_vs_epoch,seed_smrse_and_coverage_prob_tradeoff' \
-stat 'srmse' 'signedmean&' 'iter|seed&' \
-stat 'coverage_probability' '&mean|\*1000|floor' '&origin|destination||' \
-mrkr sample_name -hch sigma -c type -sz coverage_probability -v 0.5 -or asc coverage_probability -l title \
-k seed -k iter -k type -p sca -b 0 -t 1 \
-ylim 0.0 2.0 --x_discrete -xlab '(\# Epochs, Ensemble size)' -ylab 'SRMSE' -xfq 2 3 -xfq 1 1 \
-fs 4 4 -lls 8 -afs 8 -tfs 5 -nw 20

```

<!-- -fs 5 5 -ms 20 -ff pdf -tfs 14 -afs 14 -lls 18 -als 18 -->

## Figure 4
