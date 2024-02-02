import os
import os
import shutil
from tqdm import tqdm

source_dir = 'data/curated_chembl/opt_af_coords/'
destination_dir = 'data/curated_chembl/opt_af_coords/HUMAN/'

# Get the list of files in the source directory
file_list = os.listdir(source_dir)

# Move each file to the destination directory
for file_name in tqdm(file_list, desc='Moving files'):
    source_path = os.path.join(source_dir, file_name)
    destination_path = os.path.join(destination_dir, file_name)
    
    # Avoid moving the HUMAN folder into itself
    if source_path != destination_dir:
        shutil.move(source_path, destination_path)
