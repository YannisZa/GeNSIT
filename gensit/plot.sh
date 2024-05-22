clear; gensit plot spatial geoshow -x residual_mean_colsums_spatial  \
-pdd ./data/outputs/DC/comparisons/paper_figures/figure5/ \
-el np -el MathUtils \
-e residual_mean_colsums_spatial "residual_mean_colsums.to_dataframe(name='data',dim_order=['destination'])" \
-ea intensity \
-ea "intensity_mean=intensity.mean('iter',dtype='float64')" \
-ea "residual_mean_colsums=MathUtils.l2(intensity_mean,outputs.inputs.data.ground_truth_table).where(outputs.inputs.data.test_cells_mask,drop=True).sum('origin')" \
-xlab 'Longitude' -ylab 'Latitute' -at '\textsc{GMEL}' \
-fs 10 10 -ff pdf -ft 'GMEL_mean_residual' -cm bwr  \
-cm bwr -ats 18 -ylr 90 -yls 30 0 -xls 30 0 -xts 12 0 -ats 30 -nw 1;


clear; gensit plot spatial geoshow -x residual_mean_colsums_spatial  \
-pdd ./data/outputs/DC/exp1/paper_figures/figure6/ \
-el np -el MathUtils \
-e residual_mean_colsums_spatial "residual_mean_colsums.to_dataframe(name='data',dim_order=['destination','sweep']).drop(columns=['sigma','to_learn']).reset_index().set_index('destination').drop(columns=['sigma','to_learn'])" \
-ea intensity \
-ea "intensity_mean=intensity.squeeze('seed').drop_vars('seed').mean('iter',dtype='float64')" \
-ea "residual_mean_colsums=MathUtils.l2(intensity_mean,outputs.inputs.data.ground_truth_table).where(outputs.inputs.data.test_cells_mask,drop=True).sum('origin')" \
-xlab 'Longitude' -ylab 'Latitute' -at '\textsc{SIM-NN}' \
-fs 10 10 -ff pdf -ft 'SIM_NN_mean_residual' \
-cm bwr -ats 18 -ylr 90 -yls 30 0 -xls 30 0 -xts 12 0 -ats 30 -nw 1;



clear; gensit plot spatial geoshow -x residual_mean_colsums_spatial  \
-pdd ./data/outputs/DC/exp1/paper_figures/figure7/ \
-el np -el MathUtils \
-e residual_mean_colsums_spatial "residual_mean_colsums.to_dataframe(name='data',dim_order=['destination','sweep']).drop(columns=['sigma','to_learn']).reset_index().drop(columns=['sigma','to_learn']).to_json()" \
-ea table \
-ea "table_mean=table.mean('iter',dtype='float64')" \
-ea "residual_mean_colsums=MathUtils.l2(table_mean,outputs.inputs.data.ground_truth_table).where(outputs.inputs.data.test_cells_mask,drop=True).sum('origin')" \
-xlab 'Longitude' -ylab 'Latitute' -at '\textsc{GeNSIT} (Joint)' \
-fs 10 10 -ff pdf -ft 'JointTableSIM_NN_mean_residual' \
-cm bwr -ats 18 -ylr 90 -yls 30 0 -xls 30 0 -xts 12 0 -ats 30 -nw 1;