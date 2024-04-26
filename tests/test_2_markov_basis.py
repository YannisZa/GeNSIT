import pytest
import numpy as np
from copy import deepcopy
from gensit.config import Config
from gensit.utils.misc_utils import ,f_to_array,deep_updates
from gensit.contingency_table import ContingencyTable2D
from gensit.contingency_table.MarkovBasis import instantiate_markov_basis

@pytest.fixture
def default_config():
    # Import config
    config = Config("tests/test_configs/test_0_default_config.toml")
    # Check that config is valid
    return config

@pytest.fixture
def ct(default_config):
    # Build a contingency table
    ct = ContingencyTable2D(default_config)
    return ct

def test_2d_table_markov_basis_length(default_config):
    # Fix random seed
    np.random.seed(1234)

    # Generate random row and column sizes
    rowsizes = [np.random.choice(range(2,20)) for i in range(2)]
    colsizes = [np.random.choice(range(2,20)) for i in range(2)]
    # Zip them
    table_sizes = zip(rowsizes,colsizes)

    # Loop over table sizes
    for (I,J) in table_sizes:
        # Update config
        updated_config = deepcopy(default_config)
        # print(updated_config.settings)

        # Get new row and column sums
        new_total = int(np.random.randint(5,10))
        new_rowsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(I),int(I))).tolist()
        new_colsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(J),int(J))).tolist()

        # Set update
        config_update = {"inputs":{"contingency_table":{"generate":{"I":int(I),"J":int(J),"rowsums":new_rowsums,"colsums":new_colsums}}}}

        # Change generated table size
        updated_config.settings = deep_updates(d = updated_config.settings,u = config_update,overwrite = True)

        # Generate new table
        new_ct = ContingencyTable2D(updated_config)

        # Build markov base
        mb = instantiate_markov_basis(new_ct)

        # print(new_ct.shape())
        # print(len(mb))
        # print(mb)

        assert len(mb) == int(J*(J-1)*I*(I-1)/4)


def test_2d_table_markov_basis_validity(default_config):

    # Fix random seed
    # np.random.seed(1234)

    # Generate random row and column sizes
    rowsizes = [2,3]
    colsizes = [3,3]
    # Zip them
    table_sizes = zip(rowsizes,colsizes)

    # Create dummy basis functions

    # 2x3 table
    # This is admissible
    f1 = {(0,0):1,(1,1):1,(0,1):-1,(1,0):-1,(0,2):0,(1,2):0}
    # These are inadmissible
    f2 = {(0,0):1,(1,1):-1,(0,1):1,(1,0):-1,(0,2):0,(1,2):0}
    f3 = {(0,0):1,(1,1):0,(0,1):0,(1,0):0,(0,2):1,(1,2):0}
    f4 = {(0,0):0,(1,1):-1,(0,1):0,(1,0):0,(0,2):0,(1,2):-1}

    # 3x3 table
    # This is admissible
    f5 = {(0,0):1,(1,1):1,(0,1):-1,(1,0):-1,(0,2):0,(1,2):0,(2,0):0,(2,1):0,(2,2):0}
    # These are inadmissible
    f6 = {(0,0):1,(1,1):0,(0,1):0,(1,0):0,(0,2):1,(1,2):0,(2,0):-1,(2,1):0,(2,2):0}

    basis_functions = [[f1,f2,f3,f4],[f5,f6]]

    # Loop over table sizes
    for c,(I,J) in enumerate(table_sizes):
        # Update config
        updated_config = deepcopy(default_config)
        # print(updated_config.settings)

        # Get new row and column sums
        new_total = int(np.random.randint(5,10))
        new_rowsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(I),int(I))).tolist()
        new_colsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(J),int(J))).tolist()

        # Set update
        config_update = {"inputs":{"contingency_table":{"generate":{"I":int(I),"J":int(J),"rowsums":new_rowsums,"colsums":new_colsums}}}}

        # Change generated table size
        updated_config.settings = deep_updates(updated_config.settings,config_update,overwrite = True)

        # Generate new table
        new_ct = ContingencyTable2D(updated_config)

        # Build markov basis
        mb = instantiate_markov_basis(new_ct)

        for k in range(c):
            if k == 0:
                print(f_to_array(basis_functions[c][k]))
                print(mb.ct.table_summary_statistic(f_to_array(basis_functions[c][k])))
                print(np.any(mb.ct.table_summary_statistic(f_to_array(basis_functions[c][k]))))
                print(np.any(f_to_array(basis_functions[c][k])))
                assert mb.basis_function_admissible(basis_functions[c][k])
            else:
                assert not mb.basis_function_admissible(basis_functions[c][k])


def test_2d_table_all_markov_basis_validity(default_config):

    # Fix random seed
    np.random.seed(1234)

    # Generate random row and column sizes
    rowsizes = [np.random.choice(range(2,10)) for i in range(2)]
    colsizes = [np.random.choice(range(2,10)) for i in range(2)]
    # Zip them
    table_sizes = zip(rowsizes,colsizes)

    # Loop over table sizes
    for (I,J) in table_sizes:
        # Update config
        updated_config = deepcopy(default_config)

        # Get new row and column sums
        new_total = int(np.random.randint(5,10))
        new_rowsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(I),int(I))).tolist()
        new_colsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(J),int(J))).tolist()

        # Set update
        config_update = {"inputs":{"contingency_table":{"generate":{"I":int(I),"J":int(J),"rowsums":new_rowsums,"colsums":new_colsums}}}}

        # Change generated table size
        updated_config.settings = deep_updates(updated_config.settings,config_update)

        # Generate new table
        new_ct = ContingencyTable2D(updated_config)

        # Build markov base
        mb = instantiate_markov_basis(new_ct)

        # Initialise iterator
        mb_iterator = iter(mb)

        for bf in mb_iterator:
            assert mb.basis_function_admissible(bf)

        assert mb.check_markov_basis_validity()


def test_2d_table_row_margin_preserving_markov_basis_length(default_config):
    # Fix random seed
    np.random.seed(1234)

    # Generate random row and column sizes
    rowsizes = [np.random.choice(range(2,5)) for i in range(2)]
    colsizes = [np.random.choice(range(2,5)) for i in range(2)]
    # Zip them
    table_sizes = zip(rowsizes,colsizes)

    # Loop over table sizes
    for (I,J) in table_sizes:
        # Update config
        updated_config = deepcopy(default_config)
        # print(updated_config.settings)

        # Get new row and column sums
        new_total = int(np.random.randint(5,10))
        new_rowsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(I),int(I))).tolist()
        new_colsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(J),int(J))).tolist()

        # Set update
        config_update = {"inputs":{"contingency_table":{"generate":{"I":int(I),"J":int(J),"rowsums":new_rowsums,"colsums":new_colsums}}}}

        # Change generated table size
        updated_config.settings = deep_updates(d = updated_config.settings,u = config_update,overwrite = True)
        if 'outputs' in list(updated_config.settings.keys()
            del updated_config.settings['outputs']

        # Generate new table
        new_ct = ContingencyTable2D(updated_config)

        # Update margin to fix
        new_ct.constrained = 'rows'

        # Build markov base
        mb = instantiate_markov_basis(new_ct)

        for bf in iter(mb):
            assert mb.basis_function_admissible(bf)

        assert mb.check_markov_basis_validity()

def test_2d_table_column_margin_preserving_markov_basis_length(default_config):

    # Generate random row and column sizes
    rowsizes = [np.random.choice(range(2,5)) for i in range(2)]
    colsizes = [np.random.choice(range(2,5)) for i in range(2)]
    # Zip them
    table_sizes = zip(rowsizes,colsizes)


    # Loop over table sizes
    for (I,J) in table_sizes:
        # Update config
        updated_config = deepcopy(default_config)
        # print(updated_config.settings)

        # Get new row and column sums
        new_total = int(np.random.randint(5,10))
        new_rowsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(I),int(I))).tolist()
        new_colsums = np.random.multinomial(n = new_total,pvals = np.repeat(1./int(J),int(J))).tolist()

        # Set update
        config_update = {"inputs":{"contingency_table":{"generate":{"I":int(I),"J":int(J),"rowsums":new_rowsums,"colsums":new_colsums}}}}

        # Change generated table size
        updated_config.settings = deep_updates(d = updated_config.settings,u = config_update,overwrite = True)
        if 'outputs' in list(updated_config.settings.keys()
            del updated_config.settings['outputs']

        # Generate new table
        new_ct = ContingencyTable2D(updated_config)

        # Update margin to fix
        new_ct.constrained = 'columns'

        # Build markov base
        mb = instantiate_markov_basis(new_ct)

        for bf in iter(mb):
            assert mb.basis_function_admissible(bf)
    assert mb.check_markov_basis_validity()


# def test_2d_table_markov_basis_augementation(ct):
#
#     # Generate random row and column additions
#     rowsizes = [2,2,2,0,0]
#     colsizes = [3,1,0,2,0]
#
#     # Zip them
#     table_additions = zip(rowsizes,colsizes)
#
#     for (addRows,addCols) in table_additions:
#         # Build markov base
#         mb = instantiate_markov_basis(ct)
#
#         # Markov Basis can only be augmented not shrunk!!!
#         assert addRows >= 0 and addCols >= 0
#
#         # Copy current table
#         new_ct = deepcopy(ct)
#         new_ct.resize(addRows = addRows,addCols = addCols)
#
#         # Build only new markov bases
#         mb.update_and_augment(new_ct)
#
#         # print(mb)
#         # print(len(mb))
#         # print(ct.shape())
#
#         assert len(mb) == (new_ct.I*(new_ct.I-1)*new_ct.J*(new_ct.J-1))/4
#         assert mb.check_markov_basis_validity()
