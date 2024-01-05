import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def get_study_ids():
    study_ids = []

    for page_number in tqdm(range(1, 554), desc="Getting study ids"):  # Start from page 1 to 553
        url = f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies?facet.study_type=rna-seq%20of%20coding%20rna&page={page_number}"

        print(f"Retrieving page {page_number}...")
        # Send a request to the URL
        response = requests.get(url)
        print(f"Retrieved page {page_number}")

        # Check if the request was successful
        if response.status_code == 200:
            # Create a BeautifulSoup object
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all div elements with the specified attributes
            study_divs = soup.find_all('div', {'data-type': 'study'})

            for div in study_divs:
                # Extract study ID from the 'data-accession' attribute
                study_id = div.get('data-accession')
                if study_id:
                    study_ids.append(study_id)

        else:
            print(f"Failed to retrieve page {page_number}")

    return study_ids

# Example usage
study_ids = get_study_ids()
print(study_ids)
