# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
# trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# printf "\n"
# echo "Creating necessary directories"
# python ./create_directories.py
# printf "\n"

echo "Installing python requirements and multiresticodm package"
printf "\n"
pip3 install -e .

# printf "\n"
# echo "Generating stopping times"
# printf "\n"

# echo "Creating stopping times for each dataset"
# datasets=(`cat ./docs/datasets.txt`)
# noofelements=${#datasets[*]}
# Traverse the array
# counter=0
# while [ $counter -lt $noofelements ]
# do
    # python ./stopping.py -data ${datasets[$counter]}
    # echo "python ./stopping.py -data ${datasets[$counter]}"
    # counter=$(( $counter + 1 ))
# done
# printf "\n"

echo "Run tests"
printf "\n"
pytest
