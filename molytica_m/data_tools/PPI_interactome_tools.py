from molytica_m.data_tools import create_PPI_dataset
from molytica_m.data_tools import alpha_fold_tools
from tqdm import tqdm
import numpy as np
import torch
import h5py
import os
import sqlite3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("molytica_m/ml/PPI_S_model.pth", map_location=torch.device(device)).to(device)
model.eval()

def predict_PPI_value(uniprot_A, uniprot_B):
    metadata_a = create_PPI_dataset.get_metadata(uniprot_A)
    metadata_b = create_PPI_dataset.get_metadata(uniprot_B)

    x_a, edge_index_a = create_PPI_dataset.get_graph(uniprot_A)
    x_b, edge_index_b = create_PPI_dataset.get_graph(uniprot_B)

    output = model(
        torch.tensor(metadata_a, dtype=torch.float).to(device).unsqueeze(0),
        torch.tensor(metadata_b, dtype=torch.float).to(device).unsqueeze(0),
        torch.tensor(x_a, dtype=torch.float).to(device),
        torch.tensor(edge_index_a, dtype=torch.long).to(device),
        torch.tensor(x_b, dtype=torch.float).to(device),
        torch.tensor(edge_index_b, dtype=torch.long).to(device),
    )

    PPI_value = float(output.to("cpu").detach().numpy()[0][0])

    return PPI_value

def predict_PPI_batch_values(uniprot_A, uniprot_B):
    metadata_a = create_PPI_dataset.get_metadata(uniprot_A)
    metadata_b = create_PPI_dataset.get_metadata(uniprot_B)

    x_a, edge_index_a = create_PPI_dataset.get_graph(uniprot_A)
    x_b, edge_index_b = create_PPI_dataset.get_graph(uniprot_B)

    output = model(
        torch.tensor(metadata_a, dtype=torch.float).to(device).unsqueeze(0),
        torch.tensor(metadata_b, dtype=torch.float).to(device).unsqueeze(0),
        torch.tensor(x_a, dtype=torch.float).to(device),
        torch.tensor(edge_index_a, dtype=torch.long).to(device),
        torch.tensor(x_b, dtype=torch.float).to(device),
        torch.tensor(edge_index_b, dtype=torch.long).to(device),
    )

    PPI_value = float(output.to("cpu").detach().numpy()[0][0])

    return PPI_value

def predict_PPI_prob_bidirectional(A, B):
    # Your existing function
    return (predict_PPI_value(A, B) + predict_PPI_value(B, A)) / 2

def create_database():
    # Connects to the SQLite database at the specified path. It will create a new one if it doesn't exist.
    conn = sqlite3.connect('data/ppi_probs.db')
    
    # Create a cursor object using the cursor method of the connection
    cur = conn.cursor()
    
    # Create table
    cur.execute('''CREATE TABLE IF NOT EXISTS protein_interactions
                   (Protein1 TEXT, Protein2 TEXT, Probability REAL, 
                    PRIMARY KEY (Protein1, Protein2))''')
    
    # Commit the transaction
    conn.commit()
    
    # Close the connection
    conn.close()

def insert_ppi_prob(conn, uniprot_A, uniprot_B, PPI_prob):
    cur = conn.cursor()
    # Ensure the proteins are stored in a consistent order
    protein1, protein2 = sorted([uniprot_A, uniprot_B])
    cur.execute("INSERT OR REPLACE INTO protein_interactions (Protein1, Protein2, Probability) VALUES (?, ?, ?)", 
                (protein1, protein2, PPI_prob))
    conn.commit()

def get_ppi_prob(conn, uniprot_A, uniprot_B):
    cur = conn.cursor()
    # Ensure the proteins are queried in a consistent order
    protein1, protein2 = sorted([uniprot_A, uniprot_B])

    cur.execute("SELECT Probability FROM protein_interactions WHERE Protein1 = ? AND Protein2 = ?", 
                (protein1, protein2))
    result = cur.fetchone()

    if result:
        return result[0]
    else:
        return None  # Or an appropriate default value or error handling

def add_ppi_prob(uniprot_A, uniprot_B): # Add a new PPI probability to the database if it doesn't already exist
    if not get_ppi_prob(uniprot_A, uniprot_B):
        PPI_prob = predict_PPI_prob_bidirectional(uniprot_A, uniprot_B)
        insert_ppi_prob(uniprot_A, uniprot_B, PPI_prob)

def _get_ppi_prob(conn, uniprot_A, uniprot_B):
    cur = conn.cursor()
    protein1, protein2 = sorted([uniprot_A, uniprot_B])
    cur.execute("SELECT Probability FROM protein_interactions WHERE Protein1 = ? AND Protein2 = ?", 
                (protein1, protein2))
    result = cur.fetchone()
    return result[0] if result else None

def get_ppi_prob(uniprot_A, uniprot_B, db_path='data/ppi_probs.db'):
    with sqlite3.connect(db_path) as conn:
        return _get_ppi_prob(conn, uniprot_A, uniprot_B)



if __name__ == "__main__":
    create_database()

    af_uniprots = alpha_fold_tools.get_all_alphafold_uniprot_ids()["HUMAN"]
    total_iterations = len(af_uniprots) * (len(af_uniprots) + 1) / 2
    progress_bar = tqdm(total=total_iterations, desc="Overall progress")

    count = 0
    for i, uniprot_A in enumerate(af_uniprots):
        for uniprot_B in af_uniprots[i:]:
            count += 1
            progress_bar.update(1)
            if count < 0:
                continue
            add_ppi_prob(uniprot_A, uniprot_B)

    progress_bar.close()
    conn.close()
    
"""

if __name__ == "__main__":
    create_database()
    conn = sqlite3.connect('data/ppi_probs.db')

    af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()
    total_iterations = len(af_uniprots) * (len(af_uniprots) + 1) / 2
    progress_bar = tqdm(total=total_iterations, desc="Overall progress")

    count = 0
    for i in range(len(af_uniprots) - 1, -1, -1):
        uniprot_A = af_uniprots[i]
        for j in range(i, -1, -1):
            progress_bar.update(1)
            count += 1
            if count < 1774000:
                continue
            uniprot_B = af_uniprots[j]
            PPI_prob = predict_PPI_prob_bidirectional(uniprot_A, uniprot_B)
            insert_ppi_prob(conn, uniprot_A, uniprot_B, PPI_prob)
            

    progress_bar.close()
    conn.close()

"""
