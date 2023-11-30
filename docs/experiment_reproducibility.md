<!-- # Cambridge commuter LSOAs to MSOAs -->

# Experiment runs

Set `ulimit -n 50000`

## Experiment 1

```
clear; multiresticodm run ./data/inputs/configs/experiment1.toml -nw 10 -sm -et SIM_NN
```

```
clear; multiresticodm run ./data/inputs/configs/debug.toml -nw 3 -nt 12 -et SIM_NN
```

## Experiment 2

### Dependent (joint) Table and SIM

clear; multiresticodm run ./data/inputs/configs/experiment2.toml -nw 1 -sm

## Experiment 3

### NN SIM

clear; multiresticodm run ./data/inputs/configs/experiment3.toml \
 -re SIM_NN -nw 10 -sm -ln dest_attraction_ts -lf mseloss

### Independent (non-joint) Table and SIM

clear; multiresticodm run ./data/inputs/configs/experiment3.toml \
 -re NonJointTableSIM_NN -nw 10 -sm -ln dest_attraction_ts -lf mseloss

### Dependent (joint) Table and SIM

clear; multiresticodm run ./data/inputs/configs/experiment3.toml \
 -re JointTableSIM_NN -nw 10 -sm \
 -ln dest_attraction_ts -ln table_likelihood \
 -lf mseloss -lf custom

## Experiment 4

hi

## Experiment 5 (Expected loss)

Set `ulimit -n 50000`

### Dependent (joint) Table and SIM

clear; multiresticodm run ./data/inputs/configs/experiment_expected_loss.toml -et JointTableSIM_NN -nw 10 -sm

# Plots

## Experiment 2

### Figure 4

`pkill -9 -f 'multiresticodm plot'; `

-pdd /home/iz230/MultiResTICODM/data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp2/NonJointTableSIM_NN_SweepedNoise\_\_31_10_2023_09_44_49/paper_figures/ -dn cambridge_work_commuter_lsoas_to_msoas/exp2 \

```
clear; multiresticodm plot -y srmse -x 'iter&seed' -x sigma \
-dn cambridge_work_commuter_lsoas_to_msoas/exp2 -et NonJointTableSIM_NN -et JointTableSIM_NN \
-s table -s intensity -ft 'srmse_vs_epoch,seed_smrse_and_coverage_prob_tradeoff' \
-stat 'srmse' 'signedmean&' 'iter|seed&' \
-stat 'coverage_probability' '&mean|*1000|floor' '&origin|destination||' \
-mrkr sample_name -hch sigma -c type -sz coverage_probability -v 0.5 -or asc coverage_probability -l title \
-k seed -k iter -k type -p dss -b 0 -t 1 \
-ylim 0.0 2.0 --x_discrete -xlab '(\# Epochs, Ensemble size)' -ylab 'SRMSE' -xfq 2 3 -xfq 1 1  \
-fs 4 4 -lls 8 -afs 8 -tfs 5 -nw 20
```

<!-- -fs 5 5 -ms 20 -ff pdf -tfs 14 -afs 14 -lls 18 -als 18 -->

# Summaries and Metrics

## Experiment 1

```
clear; multiresticodm summarise -et SIM_NN -s intensity \
-dn cambridge_work_commuter_lsoas_to_msoas/exp1 \
-stat 'srmse' 'mean&' 'iter+seed&' \
-stat 'coverage_probability' '&mean|*100|floor' '&origin+destination||' \
-k sigma -k name -btt 'iter' 0 100 100 -nw 6 --region_mass 0.99
```

## Experiment 5 (Expected loss)

<!-- -et JointTableSIM_NN -dn cambridge_work_commuter_lsoas_to_msoas/exp5_expected_loss \ -->

<!-- -pdd /home/iz230/MultiResTICODM/data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp5_expected_loss/JointTableSIM_NN_LearnedNoise\_\_22_11_2023_20_42_35/paper_figures \ -->

```
clear; multiresticodm plot -y total -x table_steps \
-et JointTableSIM_NN -dn cambridge_work_commuter_lsoas_to_msoas/exp5_expected_loss \
-p dss -s total -ft 'total_loss_vs_table_steps' \
-stat '' 'mean&' 'iter&' \
-c title -v 0.5 -sz 20 -l title \
-k iter -b 0 -t 1 -ylim 170000 400000 -xlim 0 42 \
--x_discrete -xlab '(\# Table steps)' -ylab 'Total loss' -xfq 6 10  \
-fs 4 4 -lls 8 -afs 8 -tfs 5 -nw 6
```

```
clear; multiresticodm plot -y table_likelihood -x table_steps \
-et JointTableSIM_NN -dn cambridge_work_commuter_lsoas_to_msoas/exp5_expected_loss \
-p dss -s table_likelihood -ft 'table_loss_vs_table_steps' \
-stat '' 'mean&' 'iter&' -c title -v 0.5 -sz 20 -l title -b 0 -t 1 \
-xlim 0 42 --x_discrete -xlab '(\# Table steps)' \
-ylab 'Table loss' -xfq 6 10  -fs 4 4 -lls 8 -afs 8 -tfs 5 -nw 6

-cs title _total_constrained
```
