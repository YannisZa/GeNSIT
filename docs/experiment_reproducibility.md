# Synthetic data

Repeat the following by changing the -d argument based on the synthetic dataset and -k argument based on the ensemble size

## Direct sampling

### Unconstrained

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_2x3_N_100/ -ax '[]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et unconstrained

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_2x3_N_5000/ -ax '[]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et unconstrained

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_100/ -ax '[]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -sm -et unconstrained

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_5000/ -ax '[]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et unconstrained

### Grand total margin constrained

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_2x3_N_100/ -ax '[0, 1]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et grand_total

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_2x3_N_5000/ -ax '[0, 1]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et grand_total

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_100/ -ax '[0, 1]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -sm -et grand_total

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_5000/ -ax '[0, 1]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et grand_total

### Row margin constrained

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_2x3_N_100/ -ax '[1]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et row_margin

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_2x3_N_5000/ -ax '[1]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et row_margin

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_100/ -ax '[1]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -sm -et row_margin

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_5000/ -ax '[1]' \
-n 1000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et row_margin

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_5000/ -ax '[1]' \
-n 10000 -re exp10 -k 1000 -nw 1 -p direct_sampling -et row_margin

## Markov Basis MCMC

### Row margin constrained

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_5000/ -ax '[1]' \
 -n 10000 -re exp10 -k 1000 -nw 1 -p degree_one -et row_margin

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_5000/ -ax '[1]' \
 -n 10000 -re exp10 -k 1000 -nw 1 -p degree_higher -et row_margin

### Both margins constrained

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_5000/ -ax '[1]' -ax '[0]' \
 -n 10000 -re exp10 -k 1000 -nw 1 -p degree_one -et both_margins

multiresticodm run ./data/inputs/configs/table_convergence.toml \
 -d ./data/inputs/synthetic_33x33_N_5000/ -ax '[1]' -ax '[0]' \
 -n 10000 -re exp10 -k 1000 -nw 1 -p degree_higher -et both_margins

## Create synthetic data

clear; multiresticodm create ./data/inputs/configs/synthetic_data_generation.toml -smthd sde_solver -sn 100 -nw 4 -nt 4 -log debug
clear; multiresticodm create ./data/inputs/configs/synthetic_data_generation.toml -smthd sde_solver -sn 1 -nw 8 -nt 1 -log debug

## SIM Inference

### MCMC

clear; multiresticodm run ./data/inputs/configs/synthetic_data_learning.toml -nw 4 -nt 1 -mcmcnw 6 -re SIM_MCMC

### Neural Network

clear; multiresticodm run ./data/inputs/configs/synthetic_data_learning.toml -nw 12 -nt 1 -re SIM_NN

## Joint table and sim inference

## Summaries and Metrics

## Plots

### Log destination attraction predictions and residual plots

clear; multiresticodm plot -o ./data/outputs/synthetic \
-e SIM_NN -l dims -l noise_regime \
-p 31 -b 0 -t 1 -fs 5 5 -ms 20 -ff pdf -df dat -tfs 14 -afs 14 -lls 18 -als 18 --benchmark

# Cambridge commuter LSOAs to MSOAs

## Experiment 3

Set ulimit -n 50000

### Independent (non-joint) Table and SIM

clear; multiresticodm run ./data/inputs/configs/experiment3.toml \
 -re NonJointTableSIM_NN -nw 24 -nt 1 -sm -dev cpu

## Neural Network

### SIM only

clear; multiresticodm run ./data/inputs/configs/sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -re SIM_NN -nw 16 -nt 8 -et total_constrained

### Independent (non-joint) Table and SIM

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -re NonJointTableSIM_NN -nw 8 -nt 8 -n 1000 -et unconstrained -ax '[]' -sm -dev cpu

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -re NonJointTableSIM_NN -nw 8 -nt 8 -n 1000 -et total_constrained -ax '[0,1]' -dev cpu

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -re NonJointTableSIM_NN -nw 8 -nt 8 -n 1000 -et row_constrained -ax '[1]' -dev cpu

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -p degree_higher -re NonJointTableSIM_NN -nw 5 -nt 2 -n 1000 -et both_margin_constrained -ax '[1]' -ax '[0]' -dev cpu

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ -c cell_constraints_permuted_size_90_cell_percentage_10_constrained_axes_0_1_seed_1234.txt \
-p degree_higher -re NonJointTableSIM_NN -nw 5 -nt 8 -n 1000 -et both_margin_constrained_10%\_cells -ax '[1]' -ax '[0]' -dev cpu

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ -c cell_constraints_permuted_size_179_cell_percentage_20_constrained_axes_0_1_seed_1234.txt \
 -p degree_higher -re NonJointTableSIM_NN -nw 5 -nt 8 -n 1000 -et both_margin_constrained_20%\_cells -ax '[1]' -ax '[0]' -dev cpu

### Dependent (joint) Table and SIM

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -p degree_higher -re JointTableSIM_NN -nw 8 -nt 8 -n 1000 -et unconstrained -ax '[]' -sm -dev cpu

clear;multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -p degree_higher -re JointTableSIM_NN -nw 8 -nt 8 -n 1000 -et total_constrained -ax '[0,1]' -dev cpu

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -re JointTableSIM_NN -nw 8 -nt 8 -n 1000 -et row_constrained -ax '[1]' -dev cpu

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -p degree_higher -re JointTableSIM_NN -nw 5 -nt 8 -n 1000 -et both_margin_constrained -ax '[1]' -ax '[0]' -dev cpu

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ -c cell_constraints_permuted_size_90_cell_percentage_10_constrained_axes_0_1_seed_1234.txt \
 -p degree_higher -re JointTableSIM_NN -nw 5 -nt 8 -n 1000 -et both_margin_constrained_10%\_cells -ax '[1]' -ax '[0]' -dev cpu

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ -c cell_constraints_permuted_size_179_cell_percentage_20_constrained_axes_0_1_seed_1234.txt \
 -p degree_higher -re JointTableSIM_NN -nw 5 -nt 8 -n 1000 -et both_margin_constrained_20%\_cells -ax '[1]' -ax '[0]' -dev cpu

## Summaries and Metrics

### SRMSE

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e NonJointTableSIM_NN -e JointTableSIM_NN -e SIM_NN -m SRMSE -s table -s intensity -stat 'mean&' 'N&' -b 0 -t 1 -n 1000000 -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe SRMSEs -dev cpu -nw 12 -nt 1

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -d JointTableSIM_NN_SweepedNoise_both_margin_constrained_10%\_cells_25_09_2023_10_39_46 \
-d JointTableSIM_NN_SweepedNoise_both_margin_constrained_20%\_cells_25_09_2023_10_40_06 \
-d JointTableSIM_NN_SweepedNoise_both_margin_constrained_25_09_2023_10_36_25 \
-d NonJointTableSIM_NN_SweepedNoise_both_margin_constrained_10%\_cells_25_09_2023_10_38_14 \
-d NonJointTableSIM_NN_SweepedNoise_both_margin_constrained_20%\_cells_25_09_2023_10_39_06 \
-d NonJointTableSIM_NN_SweepedNoise_both_margin_constrained_25_09_2023_10_33_53 \
-m SRMSE -s table -s intensity -stat 'mean&' 'N&' -b 0 -t 1 -n 1000000 -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe SRMSEs -dev cpu -nw 12 -nt 12

clear; memray run -m multiresticodm -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -d JointTableSIM_NN_SweepedNoise_both_margin_constrained_10%\_cells_25_09_2023_10_39_46 \
-m SRMSE -s table -s intensity -stat 'mean&' 'N&' -b 0 -t 1 -n 1000000 -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe SRMSEs -dev cpu -nw 12 -nt 12

### SSI

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -e NonJointTableSIM_NN -e JointTableSIM_NN -m SSI -s table -s intensity -stat 'mean&' 'N&' -b 10 -t 2 -n 1000 -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe SSIs -dev cpu -nw 6

### Coverage probability

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -e NonJointTableSIM_NN -e JointTableSIM_NN -m coverage_probability -r 0.99 -s table -s intensity -stat '&mean' '&1_2' -b 10 -t 2 -n 100000 -k sigma -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe coverage_probabilities

### Markov Basis Distance (POSSIBLE SYNTAX ERRORS)

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -m edit_degree_higher_error -m edit_degree_one_error -s table -stat '&mean' '&0' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -tab table_lsoas_to_msoas.txt -fe edit_distances

### Bias

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -m p_distance -s table -s intensity -stat 'mean&X^2|sum' '0&|1_2' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe Bias2 --p_norm 0

### Variance

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -s table -s intensity -m 'none' -stat 'var&sum' '0&1_2' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -tab table_lsoas_to_msoas.txt -fe variance

### MSE

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -s table -s intensity -m p_distance -stat '&mean|sum' '&0|0_1' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -tab table_lsoas_to_msoas.txt -fe expected_error --p_norm 2

## Plots

### Isomap reduction

% Distances: chi_squared_distance, euclidean_distance

clear; multiresticodm plot -p 10 -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas \
-e JointTableSIM_MCMC \
-s table -nn 30 -dis euclidean_distance \
-et unconstrained -et row_margin -et both_margins -et both_margins_permuted_cells_10% -et both_margins_permuted_cells_20% \
-b 10000 -t 1000 -n 100 -nw 12 -l experiment_title -l noise_regime -fe table_space -ff pdf

clear; multiresticodm plot -p 10 -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas \
-e NeuralABM_HighNoise_row_margin \
-e NeuralABM_LearnedNoise_row_margin \
-e NeuralABM_LowNoise_row_margin \
-e \SIM_MCMC_LowNoise_row_margin \
-e \SIM_MCMC_HighNoise_row_margin \
-e JointTableSIM_MCMC_HighNoise_both_margins_permuted_cells_10% \
-e JointTableSIM_MCMC_HighNoise_both_margins_permuted_cells_20% \
-e JointTableSIM_MCMC_HighNoise_both_margins \
-e JointTableSIM_MCMC_HighNoise_row_margin \
-e JointTableSIM_MCMC_LowNoise_both_margins_permuted_cells_20% \
-e JointTableSIM_MCMC_LowNoise_both_margins_permuted_cells_10% \
-e JointTableSIM_MCMC_LowNoise_both_margins \
-e JointTableSIM_MCMC_LowNoise_row_margin \
-s intensity -nn 30 -dis euclidean_distance -tab table_lsoas_to_msoas.txt \
-b 10000 -t 1000 -n 100 -nw 12 -l type -l experiment_title -l noise_regime -fe table_and_intensity_space -ff pdf

clear; multiresticodm plot -p 10 -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas \
-e NeuralABM_HighNoise_row_margin \
-e NeuralABM_LearnedNoise_row_margin \
-e NeuralABM_LowNoise_row_margin \
-e \SIM_MCMC_LowNoise_row_margin \
-e \SIM_MCMC_HighNoise_row_margin \
-e JointTableSIM_MCMC_HighNoise_both_margins_permuted_cells_10% \
-e JointTableSIM_MCMC_HighNoise_both_margins_permuted_cells_20% \
-e JointTableSIM_MCMC_HighNoise_both_margins \
-e JointTableSIM_MCMC_HighNoise_row_margin \
-e JointTableSIM_MCMC_HighNoise_unconstrained \
-e JointTableSIM_MCMC_LowNoise_both_margins_permuted_cells_20% \
-e JointTableSIM_MCMC_LowNoise_both_margins_permuted_cells_10% \
-e JointTableSIM_MCMC_LowNoise_both_margins \
-e JointTableSIM_MCMC_LowNoise_row_margin \
-e JointTableSIM_MCMC_LowNoise_unconstrained \
--exclude ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp14_JointTableSIM_MCMC_HighNoise_row_margin_13_06_2023_14_03_14 \
-tab table_lsoas_to_msoas.txt -s table -s intensity -nn 30 -dis l_p_distance -emb tsne --ord '1' \
-b 10000 -t 900 -n 100 -nw 16 -nt 16 -nt 1 -l type -l experiment_title -l noise_regime -fe table_and_intensity_space -ff pdf

<!-- edit_distance_degree_one -->
<!-- chi_squared_distance -->
<!-- l_p_distance -->

### Convergence

clear;multiresticodm plot -p 02 -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIM_MCMC \
-s table -s intensity --no-benchmark -x MCMC_Iteration -no relative_l_1 -l experiment_title -l noise_regime -ff pdf \
-b 0 -t 100 -n 1000 --exclude exp6_JointTableSIM_MCMC_LowNoise_both_margins_26_05_2023

### Log destination attraction predictions and residual plots

clear; multiresticodm plot -dn cambridge_work_commuter_lsoas_to_msoas -o ./data/outputs/ \
-e SIM_NN -l type -l noise_regime \
-p 31 -b 0 -t 1 -fs 5 5 -ms 20 -ff pdf -df dat -tfs 14 -afs 14 -lls 18 -als 18 --benchmark

### Mixing

multiresticodm plot -d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/[EXPERIMENT_NAME] -p 20 \
-ff pdf -fs 5 5 -b 10000 -t 1

### Tabular

### Mean

clear; multiresticodm plot -tab table_lsoas_to_msoas.txt \
-d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp14_JointTableSIM_MCMC_HighNoise_both_margins_permuted_cells_20%\_18_05_2023_18_43_40 \
-d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp98_NeuralABM_LowNoise_row_margin_17_05_2023_15_51_11 \
-d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp5\_\SIM_MCMC_HighNoise_row_margin_06_02_2023_16_54_39 \
-s table -s intensity -stat 'signedmean|sum' '0|0' -stat 'sum' '1' -stat 'sum' '0' \
-p 40 -fs 20 7 -ff tex -mc Blues -ac Greens -ac Reds --transpose -b 10000 -fe method_best \
-csl 0.0 1.0 -tfs 14 -afs 18 -lls 18 -als 18 -mcl 1.0 291.0 -acl 267.0 849.0 -acl 343.0 11530.0#

### Cost matrix

clear; multiresticodm plot -tab table_lsoas_to_msoas.txt \
-d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp5\_\SIM_MCMC_HighNoise_row_margin_06_02_2023_16_54_39 \
-s cost_matrix -stat '0.2\*X-0.1' '|' -stat 'sum' '1' -stat 'sum' '0' \
-p 40 -fs 20 7 -ff tex -mc redgreen -ac yellowblue -ac yellowblue --transpose -b 10000 -fe cost_matrix \
-csl 0.0 1.0 -tfs 14 -afs 18 -lls 18 -als 18 -mcl -0.01 0.01 -acl -0.0001 0.0001 -acl -0.05 0.05

### Ground truth table

clear; multiresticodm plot -tab table_lsoas_to_msoas.txt -d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp14_JointTableSIM_MCMC_HighNoise_both_margins_permuted_cells_20%\_05_06_2023_12_40_31 -s ground_truth_table -stat '|' '|' -stat 'sum|' '1|' -stat 'sum|' '0|' -p 40 -fs 20 7 -ff pdf -mc yellowpurple -ac bluegreen -ac bluegreen --transpose -b 10000 -fe ground_truth -csl 0.0 1.0 -tfs 14 -afs 18 -lls 18 -als 18

#### Mean error

clear; multiresticodm plot -tab table_lsoas_to_msoas.txt \
-d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp14_JointTableSIM_MCMC_HighNoise_both_margins_permuted_cells_20%\_05_06_2023_12_40_31 \
-d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp98_NeuralABM_LowNoise_row_margin_17_05_2023_15_51_11 \
-d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp5\_\SIM_MCMC_HighNoise_row_margin_06_02_2023_16_54_39 \
-s table -s intensity -stat 'signedmean|error|sum' '0|1_2|0' -stat 'sum' '1' -stat 'sum' '0' -no relative_l_0 \
-p 40 -fs 20 7 -ff pdf -df txt -mc redgreen -ac yellowblue -ac yellowblue --transpose -b 10000 -fe method_best_error \
-csl 0.0 1.0 -tfs 14 -afs 14 -lls 14 -als 14 -mcl -0.009 0.005 -acl -0.0001 0.0001 -acl -0.05 0.05 --annotate

### Spatial

#### Low noise

multiresticodm plot -d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp6_JointTableSIM_MCMCLowNoise_best_r2_30_01_2023_16_15_58 -g ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/lsoas_to_msoas.geojson -p 41 -fs 20 10 -ff png -ac cgreen -ac cred -mc cool --annotate -s table -stat mean_variance '' '' -csl 0.0 1.0 -b 10000 -afs 26 -cfs 20 -lfs 26 -fe low_noise -lw 10 -op 0.2

multiresticodm plot -d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp6_JointTableSIM_MCMCLowNoise_best_r2_30_01_2023_16_15_58 -g ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/lsoas_to_msoas.geojson -p 41 -fs 20 10 -ff pdf -ac cgreen -ac cred -mc cblue --annotate -t posterior_table_mean_error -csl 0.5 1.0 -b 10000 -afs 26 -cfs 20 -lfs 26 -fe low_noise -no relative_l1

#### High noise

multiresticodm plot -d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp14_JointTableSIM_MCMCHighNoise_07_02_2023_16_15_43 -g ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/lsoas_to_msoas.geojson -p 41 -fs 20 10 -ff png -ac cgreen -ac cred -mc cool --annotate -s table -stat mean_variance '' '' -csl 0.0 1.0 -b 10000 -afs 26 -cfs 20 -mcl 2.7744176387786865 357.8244323730469 -acl 277.0 848.0 -acl 276.0 16340.0 -fe high_noise --no-colorbar -lw 10 -op 0.2

multiresticodm plot -d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp14_JointTableSIM_MCMCHighNoise_07_02_2023_16_15_43 -g ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/lsoas_to_msoas.geojson -p 41 -fs 20 10 -ff pdf -ac cgreen -ac cred -mc cblue --annotate -t posterior_table_mean_error -csl 0.5 1.0 -b 10000 -afs 26 -cfs 20 -mcl 7.592240947332791e-07 0.009970745312129123 -acl 0.0 0.0 -acl 0.000530141493261603 0.07746821280692036 -fe high_noise --no-colorbar -no relative_l1

# Competitive methods
