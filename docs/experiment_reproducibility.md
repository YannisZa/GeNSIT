# Cambridge commuter LSOAs to MSOAs

## Experiment 2

Set `ulimit -n 50000`

### Dependent (joint) Table and SIM

clear; multiresticodm run ./data/inputs/configs/experiment2.toml \
 -re JointTableSIM_NN -nw 10 -sm

## Experiment 3

Set `ulimit -n 50000`

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

## Experiment 5 (Expected loss)

Set `ulimit -n 50000`

### Dependent (joint) Table and SIM

clear; multiresticodm run ./data/inputs/configs/experiment_expected_loss.toml \
 -re JointTableSIM_NN -nw 10 -sm

## Plots

# Experiment 2

Set `ulimit -n 50000`

## Figure 4

-s intensity

-pdd /home/iz230/MultiResTICODM/data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp2/NonJointTableSIM_NN_SweepedNoise\_\_31_10_2023_09_44_49/paper_figures/ -dn cambridge_work_commuter_lsoas_to_msoas/exp2 \

```
clear; multiresticodm plot -y srmse -x iter -x seed \
-et NonJointTableSIM_NN -dn cambridge_work_commuter_lsoas_to_msoas/exp2 \
-s table -ft 'srmse_vs_epoch,seed_smrse_and_coverage_prob_tradeoff' \
-stat 'srmse' 'signedmean&' 'iter_seed&' \
-stat 'coverage_probability' '&mean|*1000|floor' '&origin_destination||' \
-mrkr sample_name -hch sigma -c srmse -sz coverage_probability -k seed -k iter -k type \
-p dss -b 0 -t 1 -lls 8 -xlim -1 7 --x_discrete -xlab '(# Epochs, Ensemble size)' -ylab 'SRMSE'

```

```
clear; multiresticodm plot seed iter -dn cambridge_work_commuter_lsoas_to_msoas/exp2 \
-et NonJointTableSIM_NN -s table -s intensity -ft 'seed_vs_epoch_smrse_and_coverage_prob_tradeoff' \
-stat 'srmse' 'signedmean&' 'iter_seed&' \
-stat 'coverage_probability' '&mean' '&origin_destination' \
-l type -l sample_name -l sigma -k seed -k iter -k type \
-p dss -b 0 -t 1 -nw 20 --force_reload
```

<!-- -fs 5 5 -ms 20 -ff pdf -tfs 14 -afs 14 -lls 18 -als 18 -->

## Summaries and Metrics

## Experiment 5 (Expected loss)

clear; multiresticodm summarise -o ./data/outputs/ \
-dn cambridge_work_commuter_lsoas_to_msoas/ -et JointTableSIM_NN -s table -s intensity -s loss \
-stat 'srmse' 'signedmean&' 'iter_seed&' \
-stat 'coverage_probability' '&mean|\*1000|floor' '&origin_destination||' \
-stat 'none' '&mean' '&iter' \
-b 0 -t 1 -n 1000000 -k title -k type -tab table_lsoas_to_msoas.txt -fe expected_loss_comparisons -dev cpu -nw 1
