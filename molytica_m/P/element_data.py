import pandas as pd
from tqdm import tqdm
from molytica_m.data_tools.gpt_tools import ask_gpt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('molytica_m/P/elements.csv')

# Extract the 'name' column as a list
elements_list = df['name'].tolist()

# Print the list of elements
print(elements_list)

with open("molytica_m/P/properties.txt", "r") as file:
    prompt = file.read()

print(prompt)

for element in tqdm(elements_list, desc="Loading elements data"):
    result = ask_gpt(element, prompt, "gpt-4-1106-preview", 0.2)
    print(element)
    print(result)
