<!-- # Cambridge commuter LSOAs to MSOAs -->

# Experiment runs

Set `ulimit -n 50000`

## Experiment 1

```
clear; multiresticodm run ./data/inputs/configs/experiment1_disjoint.toml -sm -et NonJointTableSIM_NN -nt 6 -nw 4
clear; multiresticodm run ./data/inputs/configs/experiment1_joint.toml -sm -et JointTableSIM_NN -nt 6 -nw 4
```

## Experiment 2

```
clear; multiresticodm run ./data/inputs/configs/experiment2_disjoint.toml -sm -et NonJointTableSIM_NN -nt 20 -nw 1
clear; multiresticodm run ./data/inputs/configs/experiment2_joint.toml -sm -et JointTableSIM_NN -nt 20 -nw 1
```

## Experiment 3

```
clear; multiresticodm run ./data/inputs/configs/experiment3_disjoint.toml -sm -et NonJointTableSIM_NN -nt 6 -nw 5
clear; multiresticodm run ./data/inputs/configs/experiment3_joint.toml -sm -et JointTableSIM_NN -nt 6 -nw 4
```

## Experiment 4

## Experiment 5 (Expected loss)

```
clear; multiresticodm run ./data/inputs/configs/experiment_expected_loss.toml -et JointTableSIM_NN -nw 15 -nt 2 -sm
```

# Summaries and Metrics

## Experiment 1

Get SRMSEs for all samples and all experiments:

```
-et JointTableSIM_NN

clear; multiresticodm summarise -ma table -ma intensity \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-et SIM_MCMC -et JointTableSIM_MCMC -et SIM_NN -et NonJointTableSIM_NN \
-stat 'srmse' 'mean&' 'iter+seed&' \
-k sigma -k name -btt 'iter' 0 100 100 \
-fe SRMSEs -nw 1 --force_reload

```

Get coverage probabilities for all samples and all experiments

```
-et JointTableSIM_MCMC -et SIM_NN -et NonJointTableSIM_NN -et JointTableSIM_NN

clear; multiresticodm summarise -ma table -ma intensity \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-et SIM_MCMC \
-stat 'coverage_probability' '&mean|\*100|rint' '&origin+destination||' \
-k sigma -k name -k type --region_mass 0.99 \
-fe CoverageProbabilities -nw 1 --force_reload

```

# Plots

## Figure 1

```
clear; multiresticodm plot -y srmse -x joint_loss \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 -et NonJointTableSIM_NN \
-ma intensity -ma table -ma dest_attraction_ts_likelihood_loss \
-stat 'srmse' 'signedmean&' 'iter|seed&' \
-e joint_loss 'dest_attraction_ts_likelihood_loss+lossfn(table,torch.log(intensity))' \
-ea dest_attraction_ts_likelihood_loss -ea table -ea intensity \
-ea lossfn=outputs.ct_mcmc.table_likelihood_loss \
-mrkr sample_name -hch sigma -c type -v 0.5 -l title \
-k seed -k iter -k type -p dss \
-btt 'iter' 0 100 100 -cs loss_name "['dest_attraction_ts_likelihood_loss']" \
-xlab 'SRMSE' -ylab 'Dest attraction likelihood loss' -ft 'srmse_vs_dest_attraction_loss' \
-fs 4 4 -lls 8 -afs 8 -tfs 5 -nw 20
```

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
-k seed -k iter -k type -p dss -b 0 -t 1 \
-ylim 0.0 2.0 --x_discrete -xlab '(\# Epochs, Ensemble size)' -ylab 'SRMSE' -xfq 2 3 -xfq 1 1 \
-fs 4 4 -lls 8 -afs 8 -tfs 5 -nw 20

```

<!-- -fs 5 5 -ms 20 -ff pdf -tfs 14 -afs 14 -lls 18 -als 18 -->

## Experiment 5 (Expected loss)

<!-- -et JointTableSIM_NN -dn cambridge_work_commuter_lsoas_to_msoas/exp5_expected_loss \ -->

<!-- -pdd /home/iz230/MultiResTICODM/data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp5_expected_loss/JointTableSIM_NN_LearnedNoise\_\_22_11_2023_20_42_35/paper_figures \ -->

```

clear; multiresticodm plot -y table_likelihood_loss -x table_steps \
-et JointTableSIM_NN -dn cambridge_work_commuter_lsoas_to_msoas/exp5_expected_loss \
-p dss -ma table_likelihood_loss -ft 'table_likelihood_loss_vs_table_steps' \
-stat '' 'mean&' 'iter&' \
-c title -v 0.5 -sz 20 -l title \
-k iter -btt 'iter' 0 100 100  -xlim 0 30 \
--x_discrete -xlab '(\# Table steps)' -ylab 'Table likelihood loss' -xfq 4 8 \
-fs 4 4 -lls 8 -afs 8 -tfs 5 -nw 6

```
