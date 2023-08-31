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

## Create comparison plot for all results

multiresticodm plot \
 -dn synthetic_2x3_N_100 -dn synthetic_2x3_N_5000 -dn synthetic_33x33_N_100 -dn synthetic_33x33_N_5000 \
 -e direct_sampling_TableSummariesMCMCConvergence -et row_margin \
 -l table_dim -l table_total -l experiment_title \
 -fe curse_of_dimensionality --no-benchmark -x MCMC_Iteration -yl -0.00 0.05 -p 00 \
 -stat 'sum' '0' -b 0 -t 10 -n 10000 -ff pdf

multiresticodm plot \
 -dn synthetic_33x33_N_5000 \
 -e direct_sampling_TableSummariesMCMCConvergence \
 -e degree_one_TableSummariesMCMCConvergence \
 -e degree_higher_TableSummariesMCMCConvergence \
 -et row_margin -et both_margins \
 -l proposal -l experiment_title \
 -fe proposal_comparison --no-benchmark -x MCMC_Iteration -p 00 \
 -stat 'sum' '0' -b 0 -t 90 -n 100000 -ff pdf \
 -exc exp10_K1000_direct_sampling_TableSummariesMCMCConvergence_row_margin_30_05_2023_14_35_30

# London Retail inference

## RSquared and LogTarget analysis

multiresticodm run ./data/inputs/configs/sim_inference_low_noise_mcmc.toml -d ./data/inputs/london_retail -cm cost_mat.txt -od P.txt -lda xd0.txt -delta 0.006122448979591836 -bm 700000 -gs 100 -re exp2

### Low noise

multiresticodm run ./data/inputs/configs/sim_inference_low_noise_mcmc.toml -d ./data/inputs/london_retail -nw 4 -delta 0.00612245 -bm 700000 -gs 100

multiresticodm run ./data/inputs/configs/sim_inference_low_noise_mcmc.toml -d ./data/inputs/london_retail \
-cm london_borough_centroid_cost_matrix.txt -nw 4 -delta 0.00612245 -bm 700000 -gs 100

### High noise

multiresticodm run ./data/inputs/configs/sim_inference_high_noise_mcmc.toml -d ./data/inputs/london_retail -nw 6 -delta 0.00612245 -bm 700000 -cm cost_mat.txt -od P.txt -lda log_destination_attraction.txt -re exp5

## RSquared and LogTarget plots

multiresticodm plot -dn london_retail -od ./data/outputs/ -e RSquaredAnalysisLowNoise -p 24 -mc RdYlGn -ms 20 -ff png
multiresticodm plot -dn london_retail -od ./data/outputs/ -e LogTargetAnalysisLowNoise -p 24 -mc RdYlGn -ms 20 -ff png

multiresticodm plot -d ./data/outputs/london_retail_exp2_RSquaredAnalysisLowNoise_14_12_2022 -p 24 -mc RdYlGn -ms 20 -ff png; multiresticodm plot -d ./data/outputs/london_retail_exp1_LogTargetAnalysisLowNoise_14_12_2022 -p 25 -mc RdYlGn -ms 20 -ff png

## SIM inference

### Low noise

multiresticodm run ./data/inputs/configs/sim_inference_low_noise_mcmc.toml -d ./data/inputs/london_retail -nw 4 -delta 0.00612245 \
-cov '0.00749674,0.00182529,0.00182529,0.00709968' -ls 50 -lss 0.02 -bm 700000 -n 50000

### High noise

multiresticodm run ./data/inputs/configs/sim_inference_high_noise_mcmc.toml -re exp5 -d ./data/inputs/london_retail -cm cost_mat.txt \
-od P.txt -lda log_destination_attraction.txt -delta 0.00612245 -kappa 1.30000005 -bm 700000 \
-cov '1.0,0.0,0.0,1.0' -ss 0.3 -ls 50 -lss 0.02 -als 10 -alss 0.1 -as 10 -nb 50 -n 100000

## Joint table and sim inference

### Low noise

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_low_noise_mcmc.toml -d ./data/inputs/london_retail -delta 0.00612245 \
-cov '0.00349674,0.00182529,0.00182529,0.00309968' -ls 100 -lss 0.01 -bm 700000 -tab '' -j 49 -n 50000 -nw 4

### High noise

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_high_noise_mcmc.toml -d ./data/inputs/london_retail -delta 0.00612245 \
-cov '1.0,0.0,0.0,1.0' -ss 0.1 -ls 50 -lss 0.02 -als 10 -alss 0.1 -as 10 -nb 50 -bm 700000 -n 1000 -nw 6 -tab '' -j 49

# Cambridge commuter LSOAs to MSOAs

## RSquared and LogTarget analysis

multiresticodm run ./data/inputs/configs/sim_inference_low_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 \
 -delta 0.012787723785166238 \
 -kappa 1.0255754475703323 \
 -gs 200 -re exp2 -fe best_r2

multiresticodm run ./data/inputs/configs/sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_euclidean_20_01_2023_clustered_facilities_sample_20x20_ripleys_k_1000_euclidean_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 \
 -delta 0.01342710997442455 \
 -kappa 1.1255754475703323 \
 -gs 200 -re exp2 -fe best_r2

### RSquared and LogTarget analysis plots

multiresticodm plot -d ./data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp2_RSquaredAnalysisLowNoise_best_r2_27_01_2023_18_11_00 -p 24 -mc RdYlGn -ms 20 -ff png
multiresticodm plot -dn cambridge_work_commuter_lsoas_to_msoas -od ./data/outputs/ -e RSquaredAnalysisLowNoise -fe best_r2_beta_rescaled -p 24 -mc RdYlGn -ms 20 -ff png

multiresticodm plot -dn cambridge_work_commuter_lsoas_to_msoas -od ./data/outputs/ -e RSquaredAnalysisLowNoise -p 24 -mc RdYlGn -ms 20 -ff png
multiresticodm plot -d ./data/outputs/cambridge_work_commuter_lsoas_exp2_RSquaredAnalysisLowNoise_16_01_2023_15_07_28 -p 24 -mc RdYlGn -ms 20 -ff png
multiresticodm plot -dn cambridge_work_commuter_lsoas_to_msoas -od ./data/outputs/ -e LogTargetAnalysisLowNoise -fe best_r2_beta_rescaled -p 25 -mc RdYlGn -ms 20 -ff png
multiresticodm plot -dn cambridge_work_commuter_lsoas_to_msoas -od ./data/outputs/ -e LogTargetAnalysisLowNoise -p 25 -mc RdYlGn -ms 20 -ff png

## Summaries and metrics

multiresticodm summarise cambridge_work_commuter_lsoas_to_msoas RSquaredAnalysisLowNoise -d 26_01_2023 -m R^2 -m fitted_alpha -m fitted_beta -m delta -m kappa -m cost_matrix -s R^2

multiresticodm summarise cambridge_work_commuter_lsoas_to_msoas RSquaredAnalysisGridSearchLowNoise -d 23_01_2023 -m delta -m kappa -m grid_size -m beta_max -m R^2 -m fitted_alpha -m fitted_beta -m fitted_scaled_beta -m cost_matrix -m datetime -s R^2

## (Joint) SIM MCMC [low noise]

### SIM MCMC [low noise]

multiresticodm run ./data/inputs/configs/sim_inference_low_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -ax '[]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -cov '0.00249674,0.0,0.0,0.0159968' \
 -ss 1.0 -alpha0 1.0 -beta0 1.0 \
 -ls 100 -lss 0.01 \
 -re exp5 -nw 1 -nt 6 -nt 1 -et grand_total \
 -n 100000 -sp 0.1

### Joint Table-SIM MCMC [low noise]

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_low_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -cov '0.00249674,0.0,0.0,0.0159968' \
 -ss 1.0 -alpha0 1.0 -beta0 1.0 \
 -ls 100 -lss 0.01 \
 -re exp6 -nw 1 -nt 6 -nt 1 -et unconstrained \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_low_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[0, 1]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -cov '0.00249674,0.0,0.0,0.0159968' \
 -ss 1.0 -alpha0 1.0 -beta0 1.0 \
 -ls 100 -lss 0.01 \
 -re exp6 -nw 1 -nt 6 -nt 1 -et grand_total \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_low_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[1]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -cov '0.00249674,0.0,0.0,0.0159968' \
 -ss 1.0 -alpha0 1.0 -beta0 1.0 \
 -ls 100 -lss 0.01 \
 -re exp6 -nw 1 -nt 6 -nt 1 -et row_margin \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_low_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[1]' -sim ProductionConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -cov '0.00249674,0.0,0.0,0.0159968' \
 -ss 1.0 -alpha0 1.0 -beta0 1.0 \
 -ls 100 -lss 0.01 \
 -re exp6 -nw 1 -nt 12 -nt 12 -et row_margin \
 -n 100000 -sp 0.05

clear; multiresticodm run ./data/inputs/configs/joint_table_sim_inference_low_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt -p degree_higher \
 -ax '[0]' -ax '[1]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -cov '0.00249674,0.0,0.0,0.0159968' \
 -ss 1.0 -alpha0 1.0 -beta0 1.0 \
 -ls 100 -lss 0.01 \
 -re exp6 -nw 1 -nt 16 -nt 16 -et both_margins \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_low_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt -p degree_higher \
 -c cell_constraints_permuted_size_90_cell_percentage_10_constrained_axes_0_1_seed_1234.txt \
 -ax '[0]' -ax '[1]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -tab0 iterative_residual_filling_solution \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -cov '0.00249674,0.0,0.0,0.0159968' \
 -ss 1.0 -alpha0 1.0 -beta0 1.0 \
 -ls 100 -lss 0.01 \
 -re exp6 -nw 1 -nt 6 -nt 1 -et both_margins_permuted_cells_10% \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_low_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt -p degree_higher \
 -c cell_constraints_permuted_size_179_cell_percentage_20_constrained_axes_0_1_seed_1234.txt \
 -ax '[0]' -ax '[1]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -tab0 iterative_residual_filling_solution \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -cov '0.00149674,0.0,0.0,0.0159968' \
 -ss 1.0 -alpha0 1.0 -beta0 1.0 \
 -ls 100 -lss 0.01 \
 -re exp6 -nw 1 -nt 6 -nt 1 -et both_margins_permuted_cells_20% \
 -n 100000 -sp 0.05

## (Joint) SIM MCMC [high noise]

### SIM MCMC [high noise]

multiresticodm run ./data/inputs/configs/sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -ax '[]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -alpha0 1.0 -beta0 1.0 \
 -ss 0.1 -cov '1.0,0.0,0.0,1.0' \
 -ls 100 -lss 0.01 \
 -als 10 -alss 0.1 -as 100 -nb 50\
 -re exp5 -nw 16 -nt 16 -nt 16 -et grand_total \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -ax '[]' -sim ProductionConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -alpha0 1.0 -beta0 1.0 \
 -ss 0.1 -cov '1.0,0.0,0.0,1.0' \
 -ls 100 -lss 0.01 \
 -als 10 -alss 0.1 -as 100 -nb 30\
 -re exp5 -nw 16 -nt 16 -nt 16 -et row_margin \
 -n 100000 -sp 0.05

### Joint Table-SIM MCMC [high noise]

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -alpha0 1.0 -beta0 1.0 \
 -ss 1.0 -cov '0.01,0.0,0.0,0.04' \
 -ls 100 -lss 0.01 \
 -als 100 -alss 0.1 -as 100 -nb 30\
 -re exp14 -nw 16 -nt 16 -nt 16 -et unconstrained \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[0, 1]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -alpha0 1.0 -beta0 1.0 \
 -ss 1.0 -cov '0.01,0.0,0.0,0.05' \
 -ls 100 -lss 0.01 \
 -als 10 -alss 0.1 -as 100 -nb 30 \
 -re exp14 -nw 12 -nt 12 -nt 12 -et grand_total \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[1]' -sim TotallyConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -tab0 iterative_residual_filling_solution \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -alpha0 1.0 -beta0 1.0 \
 -ss 1.0 -cov '0.01,0.0,0.0,0.05' \
 -ls 100 -lss 0.01 \
 -als 10 -alss 0.1 -as 100 -nb 30 \
 -re exp14 -nw 16 -nt 16 -nt 16 -et row_margin \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[1]' -sim ProductionConstrained \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -tab0 iterative_residual_filling_solution \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -alpha0 1.0 -beta0 1.0 \
 -ss 1.0 -cov '0.01,0.0,0.0,0.05' \
 -ls 100 -lss 0.01 \
 -als 10 -alss 0.1 -as 100 -nb 30 \
 -re exp14 -nw 16 -nt 12 -nt 6 -et row_margin \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[0]' -ax '[1]' -sim TotallyConstrained -p degree_higher \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -tab0 iterative_residual_filling_solution \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -alpha0 1.0 -beta0 1.0 \
 -ss 1.0 -cov '0.01,0.0,0.0,0.03' \
 -ls 100 -lss 0.01 \
 -als 10 -alss 0.1 -as 100 -nb 30 \
 -re exp14 -nw 16 -nt 16 -nt 16 -et both_margins \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[0]' -ax '[1]' -sim TotallyConstrained -p degree_higher \
 -c cell_constraints_permuted_size_90_cell_percentage_10_constrained_axes_0_1_seed_1234.txt \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -tab0 iterative_residual_filling_solution \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -alpha0 0.0 -beta0 1.0 \
 -ss 1.0 -cov '0.01,0.0,0.0,0.03' \
 -ls 100 -lss 0.01 \
 -als 10 -alss 0.1 -as 100 -nb 30 \
 -re exp14 -nw 16 -nt 1 -nt 16 -et both_margins_permuted_cells_10% \
 -n 100000 -sp 0.05

multiresticodm run ./data/inputs/configs/joint_table_sim_inference_high_noise_mcmc.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -tab table_lsoas_to_msoas.txt \
 -ax '[0]' -ax '[1]' -sim TotallyConstrained -p degree_higher \
 -c cell_constraints_permuted_size_179_cell_percentage_20_constrained_axes_0_1_seed_1234.txt \
 -od origin_demand_sum_normalised.txt \
 -lda log_destination_attraction_sum_normalised.txt \
 -cm cost_matrices/clustered_facilities_sample_20x20_20_01_2023_sample_20x20_clustered_facilities_ripleys_k_500_euclidean_points%\_prob_origin_destination_adjusted_normalised_boundary_only_edge_corrected_cost_matrix_sum_normalised.txt \
 -tab0 iterative_residual_filling_solution \
 -bm 250 -delta 0.012787723785166238 -kappa 1.0255754475703323 \
 -alpha0 0.0 -beta0 1.0 \
 -ss 1.0 -cov '0.005,0.0,0.0,0.03' \
 -ls 100 -lss 0.01 \
 -als 10 -alss 0.1 -as 100 -nb 30 \
 -re exp14 -nw 16 -nt 1 -nt 8 -et both_margins_permuted_cells_20% \
 -n 100000 -sp 0.05

## Neural Network

### SIM only

clear; multiresticodm run-nn ./data/inputs/configs/sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -od origin_demand_sum_normalised.txt \
 -dats destination_attraction_time_series_sum_normalised.txt \
 -re SIM_NN -nw 16 -nt 1 -nt 8 -et test

### Independent (non-joint) Table and SIM

clear; multiresticodm run-nn ./data/inputs/configs/joint_table_sim_inference_neural_net.toml \
 -d ./data/inputs/cambridge_work_commuter_lsoas_to_msoas/ \
 -od origin_demand_sum_normalised.txt \
 -dats destination_attraction_time_series_sum_normalised.txt \
 -re NonJointTableSIM_NN -nw 16 -nt 1 -nt 8 -et test -ax '[]'

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

multiresticodm plot -dn cambridge_work_commuter_lsoas_to_msoas -o ./data/outputs/ \
-et grand_total -et row_margin -et both_margins -et both_margins_permuted_cells_10% -et both_margins_permuted_cells_20% \
-e JointTableSIM_MCMC -l type -l noise_regime -l experiment_title \
-p 31 -b 10000 -fs 5 5 -ms 20 -ff pdf -df dat -tfs 14 -afs 14 -lls 18 -als 18 --benchmark

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

## Summaries and metrics

### SRMSE

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e NonJointTableSIM_NN -e JointTableSIM_NN -e SIM_NN -m SRMSE -s table -s intensity -stat 'mean&' '0&' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe SRMSEs

### SSI

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -m SSI -s table -s intensity -stat 'mean&' '0&' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe SSIs

### Coverage probability

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -m coverage_probability -r 0.99 -s table -s intensity -stat '&mean' '&1_2' -b 10 -t 2 -n 100000 -k sigma -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe coverage_probabilities

### Markov Basis Distance (POSSIBLE SYNTAX ERRORS)

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -m edit_degree_higher_error -m edit_degree_one_error -s table -stat '&mean' '&0' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -tab table_lsoas_to_msoas.txt -fe edit_distances

### Bias

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -m p_distance -s table -s intensity -stat 'mean&X^2|sum' '0&|1_2' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -k type -tab table_lsoas_to_msoas.txt -fe Bias2 --p_norm 0

### Variance

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -s table -s intensity -m 'none' -stat 'var&sum' '0&1_2' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -tab table_lsoas_to_msoas.txt -fe variance

### MSE

clear;multiresticodm summarise -o ./data/outputs/ -dn cambridge_work_commuter_lsoas_to_msoas -e JointTableSIMLatentMCMC -e SIMLatentMCMC -e SIM_NN -s table -s intensity -m p_distance -stat '&mean|sum' '&0|0_1' -b 10 -t 2 -n 1000 -k sigma -k experiment_title -tab table_lsoas_to_msoas.txt -fe expected_error --p_norm 2

# Competitive methods

## Neural Network

### Run models

utopya run HarrisWilson --cs Cambridge_dataset --no-eval

### Evaluate models

utopya eval HarrisWilson --cs Cambridge_dataset --po low_noise/alpha --po low_noise/beta --po high_noise/alpha --po high_noise/beta

utopya eval HarrisWilson --cs Cambridge_dataset --po low_noise/alpha --po low_noise/beta --po high_noise/alpha --po high_noise/beta --po low_noise/kappa --po high_noise/kappa

utopya eval HarrisWilson --cs Cambridge_dataset --po low_noise/alpha --po low_noise/beta --po high_noise/alpha --po high_noise/beta --po low_noise/kappa --po high_noise/kappa outputs/HarrisWilson/230517-112911_Cambridge_dataset

utopya eval HarrisWilson --cs Cambridge_dataset --po destination_sizes_predictions
