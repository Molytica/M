from ftplib import FTP
import os


def download_all():
    try:
        # Define the directory where files will be saved
        save_directory = 'data/pubchem_assays'

        # Ensure the save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Function to download a file
        def grabFile(filename):
            local_filename = os.path.join(save_directory, filename)
            with open(local_filename, 'wb') as local_file:
                ftp.retrbinary('RETR ' + filename, local_file.write)

        # Connect to the FTP server
        ftp = FTP('ftp.ncbi.nlm.nih.gov')
        ftp.login()  # Anonymous login

        # Change to the target directory for JSON files
        ftp.cwd('/pubchem/Bioassay/JSON/')

        # Get a list of files in the directory
        files = ftp.nlst('*.zip')  # Adjust pattern as needed

        # Download each file
        for file in files:
            
            if not os.path.exists(os.path.join(save_directory, file)):
                print(f'Downloading {file} to {save_directory}...')
                grabFile(file)
                print(f'Finished downloading {file}.')
            else:
                print(f'File {file} already exists in {save_directory}.')

        # Close the connection
        ftp.quit()
        print('All files have been downloaded to the pubchem_assays folder.')
    except Exception as e:
        download_all()

if __name__ == "__main__":
    download_all()