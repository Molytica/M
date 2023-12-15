import sqlite3, sys
from molytica_m.data_tools.graph_tools import graph_tools
from molytica_m.data_tools import create_PPI_dataset
from torch_geometric.data import Data, Batch
from molytica_m.ml.iP_B_model import ProteinModulationPredictor
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace 'your_chembl_database_file.db' with the path to your ChEMBL SQLite file
database_path = 'data/chembl/smiles_alphafold_v4_human_uniprot_chembl_bioactivities.db'

# Connect to the SQLite database
conn = sqlite3.connect(database_path)

# Create a cursor object
cursor = conn.cursor()
batch_size = 30
n_IC50_EC50 = 878560
n_batches = int(n_IC50_EC50 / batch_size) + 1


model = ProteinModulationPredictor().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for x in range(0, n_IC50_EC50, batch_size):
    query = "SELECT * FROM bioactivities WHERE activity_unit = 'nM' AND (activity_type = 'IC50' OR activity_type = 'EC50') LIMIT {} OFFSET {};".format(batch_size, x)  # Replace 'your_table_name' with the actual table name
    cursor.execute(query)

    prot_metadatas = []
    mol_metadatas = []
    mol_Gs = []
    prot_Gs = []

    labels = []

    pct_yield = 0

    # Fetch and print the results
    rows = cursor.fetchall()
    for row in rows:
        smiles = row[0]
        uniprot_id = row[1]
        IC50 = 39900.0 #90th percentile value (high = low inhibition potency)
        EC50 = 26302.4 #90th percentile value (high = low effect potency)
        mol_metadata = np.array([element for element in row[6:] if type(element) != str], dtype=float)
        np.nan_to_num(mol_metadata, copy=False)

        if row[2] == 'IC50':
            IC50 = row[4]
        else:
            EC50 = row[4]

        mol_features, mol_csr_matrix = None, None
        try:
            mol_features, mol_csr_matrix = graph_tools.get_raw_graph_from_smiles_string(smiles)
        except Exception as e:
            continue

        prot_features, prot_edge_index = create_PPI_dataset.get_graph_raw(uniprot_id)

        mol_edge_index_list = np.vstack(mol_csr_matrix.nonzero())

        mol_metadatas.append(mol_metadata)
        prot_metadatas.append(create_PPI_dataset.get_metadata(uniprot_id))

        mol_Gs.append(Data(x=torch.Tensor(mol_features), edge_index=torch.tensor(mol_edge_index_list, dtype=torch.long)))
        prot_Gs.append(Data(x=torch.Tensor(prot_features), edge_index=torch.tensor(prot_edge_index, dtype=torch.long)))

        labels.append([EC50, IC50])

        pct_yield += 1
    
    pct_yield = pct_yield / len(rows)
    print(f"Pct yield: {pct_yield}")

    prot_metadatas = np.array(prot_metadatas)
    mol_metadatas = np.array(mol_metadatas)
    prot_metadatas = torch.tensor(prot_metadatas, dtype=torch.float).to(device)
    mol_metadatas = torch.tensor(mol_metadatas, dtype=torch.float).to(device)

    batch_molss = Batch.from_data_list(mol_Gs).to(device)
    batch_prots = Batch.from_data_list(prot_Gs).to(device)

    outputs = model(prot_metadatas, mol_metadatas, batch_prots.x, batch_prots.edge_index, batch_prots.batch, batch_molss.x, batch_molss.edge_index, batch_molss.batch)

    labels = torch.tensor(np.array(labels), dtype=torch.float).to(device)

    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
    print(loss)

# Close the connection
conn.close()