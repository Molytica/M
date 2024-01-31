#!/bin/bash


# Activate Conda environment
conda activate M
# Change to the correct directory
cd /home/oliver/M/

# Execute the Python script with the provided start_id
python molytica_m/QSAR/start_id.py $1

# Keep the terminal open to view output or errors
echo 'Script finished. Press enter to exit.'
read
