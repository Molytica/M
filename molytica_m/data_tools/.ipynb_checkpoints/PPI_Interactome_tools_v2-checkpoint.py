from molytica_m.data_tools import create_PPI_dataset
from molytica_m.data_tools import alpha_fold_tools
from tqdm import tqdm
import torch
import h5py
import os
import sqlite3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("molytica_m/ml/PPI_S_model.pth", map_location=torch.device(device)).to(device)
model.eval()
af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

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

def predict_PPI_prob_bidirectional(A, B):
    # Your existing function
    return (predict_PPI_value(A, B) + predict_PPI_value(B, A)) / 2

def get_db_index(uniprot_A, uniprot_B, entries_per_db=1000000):
    """Calculate the database index based on the position of uniprots."""
    # Efficient calculation of the entry count
    index_A = af_uniprots.index(uniprot_A)
    index_B = af_uniprots.index(uniprot_B)

    # Calculate entry count more efficiently
    entry_count = index_A * (index_A + 1) // 2
    if index_B >= index_A:
        entry_count += index_B - index_A

    return entry_count // entries_per_db


def get_ppi_prob(uniprot_A, uniprot_B, db_prefix='data/ppi_probs_multi_db/ppi_probs_'):
    """Fetch the PPI probability from the appropriate database."""
    db_index = get_db_index(uniprot_A, uniprot_B)
    db_path = f"{db_prefix}{db_index}.db"

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        protein1, protein2 = sorted([uniprot_A, uniprot_B])
        cur.execute(f"SELECT Probability FROM protein_interactions_{db_index} WHERE Protein1 = ? AND Protein2 = ?", 
                    (protein1, protein2))
        result = cur.fetchone()
        return result[0] if result else None


def create_database(table_suffix):
    conn = sqlite3.connect(f'data/ppi_probs_multi_db/ppi_probs_{table_suffix}.db')
    cur = conn.cursor()
    cur.execute(f'''CREATE TABLE IF NOT EXISTS protein_interactions_{table_suffix}
                   (Protein1 TEXT, Protein2 TEXT, Probability REAL, 
                    PRIMARY KEY (Protein1, Protein2))''')
    conn.commit()
    conn.close()

def insert_ppi_prob(conn, uniprot_A, uniprot_B, PPI_prob, counter, table_suffix):
    cur = conn.cursor()
    protein1, protein2 = sorted([uniprot_A, uniprot_B])
    cur.execute(f"INSERT OR REPLACE INTO protein_interactions_{table_suffix} (Protein1, Protein2, Probability) VALUES (?, ?, ?)", 
                (protein1, protein2, PPI_prob))
    conn.commit()
    counter += 1
    if counter >= 1000000:
        counter = 0
        table_suffix += 1
        create_database(table_suffix)
        conn.close()
        conn = sqlite3.connect(f'data/ppi_probs_multi_db/ppi_probs_{table_suffix}.db')
    return conn, counter, table_suffix

if __name__ == "__main__":
    table_suffix = 0
    create_database(table_suffix)
    conn = sqlite3.connect(f'data/ppi_probs_multi_db/ppi_probs_{table_suffix}.db')
    counter = 0

    total_iterations = len(af_uniprots) * (len(af_uniprots) + 1) / 2
    progress_bar = tqdm(total=total_iterations, desc="Overall progress")

    for i, uniprot_A in enumerate(af_uniprots):
        for uniprot_B in af_uniprots[i:]:  # Start from the current index of uniprot_A
            PPI_prob = predict_PPI_prob_bidirectional(uniprot_A, uniprot_B)
            conn, counter, table_suffix = insert_ppi_prob(conn, uniprot_A, uniprot_B, PPI_prob, counter, table_suffix)
            progress_bar.update(1)

    progress_bar.close()
    conn.close()

