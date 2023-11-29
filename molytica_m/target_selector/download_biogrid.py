import requests
import zipfile
import os

def download_and_unzip(url, download_path, extract_path):
    """
    Download a ZIP file from the given URL and unzip it to the specified path.
    """
    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        with open(download_path, "wb") as file:
            file.write(response.content)

        # Unzip the file
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return "Download and extraction successful."
    else:
        return "Error: Unable to download the file."

# URL of the ZIP file
url = "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.227/BIOGRID-ALL-4.4.227.tab3.zip"

# Path to save the downloaded ZIP file
download_path = "BIOGRID-ALL-4.4.227.tab3.zip"

# Path to extract the contents of the ZIP file
extract_path = "unzipped_biogrid"

# Ensure the extraction path exists
os.makedirs(extract_path, exist_ok=True)

# Download and unzip the file
result = download_and_unzip(url, download_path, extract_path)
print(result)