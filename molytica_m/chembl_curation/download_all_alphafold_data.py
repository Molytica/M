import ftplib
import os
import urllib.parse
import logging
from tqdm import tqdm

def download_alphafold_data(destination_folder="data/curated_chembl/all_alpha_fold_data"):
    # Ensure the destination directory exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # URL of the FTP directory for AlphaFold data
    url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/"

    # Parse the URL to extract the FTP host and path
    parsed_url = urllib.parse.urlparse(url)
    ftp_host = parsed_url.hostname
    ftp_path = parsed_url.path

    # Function to download files from an FTP folder
    def download_ftp_folder(ftp_host, ftp_path, destination):
        try:
            with ftplib.FTP(ftp_host) as ftp:
                ftp.login()  # Log in to the FTP server
                ftp.cwd(ftp_path)  # Change to the desired directory
                items = ftp.nlst()

                for item in tqdm(items, desc="Downloading files", unit="file"):
                    local_filename = os.path.join(destination, item)

                    # Check if file already exists to avoid re-downloading
                    if not os.path.exists(local_filename):
                        with open(local_filename, 'wb') as file:
                            ftp.retrbinary('RETR ' + item, file.write)
                        logging.info(f"Downloaded {item}")
                    else:
                        logging.info(f"File {item} already exists, skipping.")

        except ftplib.all_errors as e:
            logging.error(f"FTP error: {e}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

    # Download files
    download_ftp_folder(ftp_host, ftp_path, destination_folder)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    download_alphafold_data()

if __name__ == "__main__":
    main()
