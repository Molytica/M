import pandas as pd
import sqlite3

df_vals = pd.read_table('molytica_m/data_tools/idmapping_uniprot_chembl.tsv').values
id_map_chembl_to_uniprot = {}
tid_to_af_uniprot_dict = {}

for val in df_vals:
    chembl = val[1]
    uniprot = val[0]
    id_map_chembl_to_uniprot[chembl] = uniprot

def chembl_to_af_uniprot(chembl_id):
    try:
        return id_map_chembl_to_uniprot[chembl_id]
    except:
        return None

def tid_to_af_uniprot(tid):
    try:
        return tid_to_af_uniprot_dict[tid]
    except:
        return None

def generate_index_dictionaries(database_path='data/chembl/chembl_33.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)

    # Create a cursor object
    cursor = conn.cursor()

    # Execute a query (example: selecting 10 rows from a table)
    query = "SELECT * FROM target_dictionary;"  # Replace 'your_table_name' with the actual table name
    cursor.execute(query)

    # Fetch and print the results
    rows = cursor.fetchall()
    for row in rows:
        chembl = row[5]
        tid = row[0]
        af_uniprot = chembl_to_af_uniprot(chembl)
        
        if af_uniprot:
            tid_to_af_uniprot_dict[tid] = af_uniprot

    # Close the connection
    conn.close()