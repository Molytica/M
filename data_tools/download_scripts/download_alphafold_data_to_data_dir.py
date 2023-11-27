import os
import requests
import tarfile
from tqdm import tqdm

def download_file(url, target_path, chunk_size=1024):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(target_path, 'wb') as file, tqdm(
        desc=target_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def extract_tar_file(file_path, target_dir):
    try:
        with tarfile.open(file_path, 'r') as tar, tqdm(
            desc="Extracting",
            total=len(tar.getmembers()),
            unit='file'
        ) as bar:
            for member in tar:
                tar.extract(member, path=target_dir)
                bar.update(1)
    except Exception as e:
        print(f"Error occurred while extracting: {e}")

def download_and_extract(url, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = url.split('/')[-1]
    file_path = os.path.join(target_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        download_file(url, file_path)
    else:
        print("File {file_name} already exists...")

    print(f"Extracting {file_name}...")
    extract_tar_file(file_path, target_dir)

if __name__ == "__main__":
    url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar"
    download_and_extract(url, "data/AlphaFoldData")
