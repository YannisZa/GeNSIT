"""
To generate random stopping times from P(K > k) = 1./k^1.1
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fix random seed
np.random.seed(889)

# Get current working directory and project root directory
def get_project_root():
    """ Returns project's root working directory (entire path).

    Returns
    -------
    string
        Path to project's root directory.

    """
    # Get current working directory
    cwd = os.getcwd()
    # Remove all children directories
    rd = os.path.join(cwd.split('data-augmentation-spatial-interaction-models/', 1)[0])
    # Make sure directory ends with project's name
    if not rd.endswith('data-augmentation-spatial-interaction-models'):
        rd = os.path.join(rd,'data-augmentation-spatial-interaction-models/')

    return rd

# Get project directory
wd = get_project_root()

# Append project root directory to path
sys.path.append(wd)

# Parse arguments from command line
parser = argparse.ArgumentParser(description='Script to create stopping times for approximating the inverse of z(\theta) by truncating an infinite series')
parser.add_argument("-data", "--dataset_name",nargs='?',type = str,choices=['commuter_borough','commuter_ward','retail','transport','synthetic'],default = 'synthetic',
                    help="Name of dataset (this is the directory name in data/inputs)")
parser.add_argument("-n", "--n",nargs='?',type = int,default = 20000,
                    help="Number of stopping times to create.")
args = parser.parse_args()

# Get dataset name
dataset = args.dataset_name
# Get number of stopping times
nums = np.empty(args.n)

for i in range(args.n):
    N = 1
    k_pow = 1.1
    u = np.random.uniform(0, 1)
    while(u < np.power(N+1, -k_pow)):
        N += 1
    nums[i] = N

np.savetxt(os.path.join(wd,f"data/inputs/{dataset}/stopping_times.txt"), nums)
print('Done. Stopping times saved to',os.path.join(wd,f"data/inputs/{dataset}/stopping_times.txt"))
