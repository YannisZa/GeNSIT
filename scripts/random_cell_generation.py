import os
import numpy as np

from multiresticodm.utils import write_txt
from multiresticodm.contingency_table import ContingencyTable2D

# Specify path to table
table_filenames = ['./data/inputs/cambridge_work_commuter_lsoas_to_msoas/table_lsoas_to_msoas.txt',
                   './data/inputs/synthetic_33x33_N_5000/table.txt']

# Select settings to pass to contingency table
settings = {"constraints":{"axes":[[0],[1]]},"seed":1234,"sparse_margins":False}

for table_filename in table_filenames:
    # Number of cells you wish to fix
    for percentage_of_cells in [10,20]:
        # Load table
        table = np.loadtxt(table_filename).astype('int32')

        # Choose cell sampling method
        sampling_method = 'permuted'

        # Instantiate table
        ct = ContingencyTable2D(table=table,**settings)

        # Get total number of cells
        Ncells = np.prod(ct.table.shape)


        if sampling_method == 'permuted':
            # Permute cells
            permuted_cells = np.random.RandomState(seed=settings['seed']).permutation(ct.cells)
            # Get first X percentage of cells
            constrained_cells = permuted_cells[:int(np.round((percentage_of_cells*Ncells)/100))]
            # Convert to tuples
            constrained_cells = [tuple(sublist) for sublist in constrained_cells.tolist()]

        # elif sampling_method == ''

        # Update cell constraints
        settings['constraints']['cells'] = constrained_cells

        # Instantiate table with new constraints
        ct = ContingencyTable2D(table=table,**settings)

        # Print constraint table
        print(ct.constraint_table(with_margins=True))


        print(f"{len(constrained_cells)} cells selected")
        print(f"{len(ct.cells)} cells need to be learned")

        print("Minimum number of cells to be learned")
        learned_cells_ax0 = (ct.constraint_table(with_margins=False) < 0).sum(axis=0)
        learned_cells_ax1 = (ct.constraint_table(with_margins=False) < 0).sum(axis=1)
        print(f"ax = 0 | min = {np.min(learned_cells_ax0)}, argmin = {np.argmin(learned_cells_ax0)+1}")
        print(f"ax = 1 | min = {np.min(learned_cells_ax1)}, argmin = {np.argmin(learned_cells_ax1)+1}")

        # Define filename
        filename = f"cell_constraints" + \
                f"_{sampling_method}" + \
                f"_size_{len(constrained_cells)}" + \
                f"_cell_percentage_{percentage_of_cells}" + \
                f"_constrained_axes_{'_'.join([str(ax) for ax in settings['constraints']['axes']])}" + \
                f"_seed_{settings['seed']}"
        # Define filepath
        dataset_path = os.path.abspath(os.path.join(table_filename, os.pardir))

        # Save txt
        write_txt(constrained_cells,os.path.join(dataset_path,filename))