#!/usr/bin/env bash

# Read cmd arguments
experiment=${1:-""}
dataset=${2:-""}

if [ "$experiment" != "" ] && [ "$dataset" != "" ]; then
  echo "Looking for $experiment in remote repository"
  if [ -d "./data/outputs/$dataset/$experiment" ] && [ ! -L "./data/outputs/$dataset/$experiment" ]
  then
    echo "Experiment data found. Producting plots."
    # Create local directory
    mkdir -p ./data/outputs/$dataset/$experiment/figures/

    # Produce parameter mixing plot
    gensit plot -d ./data/outputs/$dataset/$experiment -p 20 -b 5000 -xfq 5 -fs 15 5 -ff png
    # Produce parameter histogram
    gensit plot -d ./data/outputs/$dataset/$experiment -p 22 -b 5000 -nb 500 -ff png
    # Produce parameter 2d contour
    gensit plot -d ./data/outputs/$dataset/$experiment -p 21 -b 5000 -fs 5 5 -ms 10 -ff png
    # Produce parameter autocorrelation plot
    gensit plot -d ./data/outputs/$dataset/$experiment -p 23 -b 5000 -fs 10 5 -nb 200 --benchmark -ff png
    # Produce log destination attraction predictions and residual plots
    gensit plot -d ./data/outputs/$dataset/$experiment -p 31 -p 32 -b 5000 -fs 5 5 -ms 20 -ff png
    # Produce posterior table mean convergence to ground truth
    gensit plot -d ./data/outputs/$dataset/$experiment -p 02 -b 0 -fs 10 10 -no relative_l_1 -ff png

  else
    echo "Directory for $experiment not found in remote server."
  fi
elif [ "$experiment" == "" ]; then
    echo "Experiment id not provided"
elif [ "$dataset" == "" ]; then
    echo "Dataset not provided"
fi
